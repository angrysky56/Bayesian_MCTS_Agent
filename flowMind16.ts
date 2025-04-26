import * as fs from 'fs/promises';
import * as fsSync from 'fs';
import * as path from 'path';
import * as os from 'os';
import * as readline from 'readline';
import { v4 as uuidv4 } from 'uuid';
import chalk from 'chalk';
import { WebSocket, WebSocketServer } from 'ws';
import { ChatOllama } from "@langchain/community/chat_models/ollama";
import { OllamaEmbeddings } from "@langchain/community/embeddings/ollama";
import { BaseChatModel } from "@langchain/core/language_models/chat_models";
import { Embeddings } from "@langchain/core/embeddings";
import { HumanMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document } from "@langchain/core/documents";
import { FaissStore } from "@langchain/community/vectorstores/faiss";
import { IncomingMessage } from 'http';

// --- Configuration ---
const DATA_DIR = path.join(os.homedir(), '.flowmind');
const STATE_FILE = path.join(DATA_DIR, 'flowmind_state.json');
const VECTOR_STORE_DIR = path.join(DATA_DIR, 'vector-store');
const OLLAMA_BASE_URL = process.env.OLLAMA_HOST || "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'hf.co/mradermacher/phi-4-GGUF:Q4_K_S'; // Updated default model
const OLLAMA_EMBEDDING_MODEL = process.env.OLLAMA_EMBED_MODEL || OLLAMA_MODEL;
const WORKER_INTERVAL = 2000; // ms
const SAVE_DEBOUNCE = 5000; // ms
const MAX_RETRIES = 2;
const DEFAULT_BELIEF_POS = 1.0;
const DEFAULT_BELIEF_NEG = 1.0;
const SHORT_ID_LEN = 6;
const WS_PORT = 8080;
const LOG_LEVEL = process.env.LOG_LEVEL || 'info';
const STUCK_THRESHOLD_MS = 5 * 60 * 1000; // 5 minutes
const TICK_INTERVAL_MS = 60 * 1000; // 1 minute
const CLIENT_REQUEST_TIMEOUT = 15000; // ms
const RECONNECT_DELAY = 5000; // ms

// --- Types ---
enum Status { PENDING = 'PENDING', ACTIVE = 'ACTIVE', WAITING = 'WAITING', DONE = 'DONE', FAILED = 'FAILED' }

enum Type {
    INPUT = 'INPUT', GOAL = 'GOAL', STRATEGY = 'STRATEGY', OUTCOME = 'OUTCOME',
    QUERY = 'QUERY', USER_PROMPT = 'USER_PROMPT', SYSTEM = 'SYSTEM',
    FACT = 'FACT', LOG = 'LOG',
    ATOM = "ATOM"
}

type TaskStatus = 'RUNNING' | 'PAUSED';
type Term = Atom | Variable | Struct | List;

interface Atom { kind: 'Atom'; name: string; }
interface Variable { kind: 'Variable'; name: string; }
interface Struct { kind: 'Struct'; name: string; args: Term[]; }
interface List { kind: 'List'; elements: Term[]; }

interface BeliefData { pos: number; neg: number; }

type WaitingCondition =
    | string // Waiting for another thought ID
    | { type: 'time', timestamp: number }
    | { type: 'event', pattern: string }; // Future: Wait for specific event pattern

interface Metadata {
    rootId?: string;
    parentId?: string;
    ruleId?: string;
    agentId?: string;
    created?: string;
    modified?: string;
    priority?: number;
    error?: string;
    promptId?: string;
    tags?: string[];
    feedback?: any[];
    embedded?: string;
    suggestions?: string[];
    retries?: number;
    waitingFor?: WaitingCondition;
    responseTo?: string;
    provenance?: string;
    taskStatus?: TaskStatus;
    [key: string]: any; // Allow flexible metadata
}

interface Thought {
    id: string;
    type: Type;
    content: Term;
    belief: Belief;
    status: Status;
    metadata: Metadata;
}

interface Rule {
    id: string;
    pattern: Term;
    action: Term;
    belief: Belief;
    metadata: {
        priority?: number;
        description?: string;
        provenance?: string;
        created?: string;
        modified?: string;
    };
}

interface MemoryEntry {
    id: string;
    embedding?: number[]; // Usually not populated after retrieval
    content: string;
    metadata: { created: string; type: string; sourceId: string; };
    score?: number; // Retrieval score
}

interface AppState { thoughts: Record<string, any>; rules: Record<string, any>; }

interface ToolContext {
    thoughts: Thoughts;
    tools: Tools;
    rules: Rules;
    memory: Memory;
    llm: LLM;
    prompts: Prompts;
    engine: Engine;
}

interface Tool {
    name: string;
    description: string;
    execute(actionTerm: Struct, context: ToolContext, trigger: Thought): Promise<Term | null>;
}

interface SystemControl {
    startProcessing: () => void;
    pauseProcessing: () => void;
    stepProcessing: () => Promise<number>;
    shutdown: () => Promise<void>;
    getStatus: () => { isRunning: boolean };
}

type MatchResult = { rule: Rule; bindings: Map<string, Term> };
type ActionResult = { success: boolean; error?: string; finalStatus?: Status };
type FallbackResult = ActionResult;
type LogLevel = 'debug' | 'info' | 'warn' | 'error';
type WsMessage = { type: string; payload?: any; success?: boolean; requestId?: string; [key: string]: any };

// --- Utilities ---
const generateId = (): string => uuidv4();
const shortId = (id: string | undefined): string => id ? id.substring(0, SHORT_ID_LEN) : '------';
const sleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms));
const LOG_LEVELS: Record<LogLevel, number> = { debug: 0, info: 1, warn: 2, error: 3 };
const CURRENT_LOG_LEVEL = LOG_LEVELS[LOG_LEVEL as LogLevel] ?? LOG_LEVELS.info;

const log = (level: LogLevel, message: string, ...args: any[]) => {
    if (LOG_LEVELS[level] >= CURRENT_LOG_LEVEL) {
        const colorMap: Record<LogLevel, chalk.ChalkFunction> =
            { debug: chalk.gray, info: chalk.blue, warn: chalk.yellow, error: chalk.red };
        console.log(colorMap[level](`[${level.toUpperCase()}] ${message}`), ...args);
    }
};

function debounce<T extends (...args: any[]) => any>(func: T, wait: number): T & {
    cancel: () => void; flush: () => ReturnType<T> | undefined;
} {
    let timeout: NodeJS.Timeout | null = null;
    let lastArgs: Parameters<T> | null = null;
    let lastThis: any = null;
    let result: ReturnType<T> | undefined;
    const debounced = (...args: Parameters<T>): void => {
        lastArgs = args;
        lastThis = this;
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
    const later = () => {
        timeout = null;
        if (lastArgs) result = func.apply(lastThis, lastArgs);
        lastArgs = null;
        lastThis = null;
    };
    debounced.cancel = () => {
        if (timeout) clearTimeout(timeout);
        timeout = null;
        lastArgs = null;
        lastThis = null;
    };
    debounced.flush = () => {
        if (timeout) clearTimeout(timeout);
        later();
        return result;
    };
    return debounced as T & { cancel: () => void; flush: () => ReturnType<T> | undefined; };
}

const safeJsonParse = <T>(json: string | null | undefined, defaultValue: T): T => {
    if (!json) return defaultValue;
    try { return JSON.parse(json); }
    catch { return defaultValue; }
};

const isValidUuid = (id: string): boolean =>
    /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/.test(id);

// --- Core Logic Classes ---

class Belief implements BeliefData {
    constructor(public pos: number = DEFAULT_BELIEF_POS, public neg: number = DEFAULT_BELIEF_NEG) { }
    score(): number { return (this.pos + 1) / (this.pos + this.neg + 2); } // Laplace smoothing
    update(success: boolean): void { success ? this.pos++ : this.neg++; }
    toJSON(): BeliefData { return { pos: this.pos, neg: this.neg }; }
    static fromJSON(data: BeliefData | Belief | undefined): Belief {
        if (!data) return new Belief();
        return data instanceof Belief ? data : new Belief(data.pos ?? DEFAULT_BELIEF_POS, data.neg ?? DEFAULT_BELIEF_NEG);
    }
    static DEFAULT = new Belief();
}

// Term Constructors (Syntactic Sugar)
const Atom = (name: string): Atom => ({ kind: 'Atom', name });
const Variable = (name: string): Variable => ({ kind: 'Variable', name });
const Struct = (name: string, args: Term[]): Struct => ({ kind: 'Struct', name, args });
const List = (elements: Term[]): List => ({ kind: 'List', elements });

namespace Terms {
    export function format(term: Term | null | undefined, useColor: boolean = true): string {
        if (!term) return useColor ? chalk.grey('null') : 'null';
        const c = (color: chalk.ChalkFunction, text: string) => useColor ? color(text) : text;
        switch (term.kind) {
            case 'Atom': return c(chalk.green, term.name);
            case 'Variable': return c(chalk.cyan, `?${term.name}`);
            case 'Struct': return `${c(chalk.yellow, term.name)}(${term.args.map(t => format(t, useColor)).join(', ')})`;
            case 'List': return `[${term.elements.map(t => format(t, useColor)).join(', ')}]`;
            default: const exhaustiveCheck: never = term; return c(chalk.red, 'invalid_term');
        }
    }

    export function toString(term: Term): string {
        switch (term.kind) {
            case 'Atom': return term.name;
            case 'Variable': return `?${term.name}`;
            case 'Struct': return `${term.name}(${term.args.map(toString).join(',')})`;
            case 'List': return `[${term.elements.map(toString).join(',')}]`;
            default: const exhaustiveCheck: never = term; return '';
        }
    }

    export function fromString(input: string): Term {
        input = input.trim();
        if (input.startsWith('?') && input.length > 1 && !input.includes('(') && !input.includes('[') && !input.includes(',')) return Variable(input.substring(1));

        const parseNested = (str: string, open: string, close: string): Term[] => {
            const items: Term[] = [];
            let currentItem = '';
            let level = 0;
            for (let i = 0; i < str.length; i++) {
                const char = str[i];
                if (char === open || char === '(' || char === '[') level++;
                else if (char === close || char === ')' || char === ']') level--;

                if (char === ',' && level === 0) {
                    if (currentItem.trim()) items.push(fromString(currentItem.trim()));
                    currentItem = '';
                } else { currentItem += char; }
            }
            if (currentItem.trim()) items.push(fromString(currentItem.trim()));
            return items;
        };

        const structMatch = input.match(/^([a-zA-Z0-9_]+)\((.*)\)$/s); // Use 's' flag for multiline content within ()
        if (structMatch) return Struct(structMatch[1], parseNested(structMatch[2], '(', ')'));

        const listMatch = input.match(/^\[(.*)]$/s); // Use 's' flag for multiline content within []
        if (listMatch) return List(parseNested(listMatch[1], '[', ']'));

        // Allow more complex atoms, but log if it looks like a malformed struct/list
        if (/^[a-zA-Z0-9_:\-\/. ]+$/.test(input) || input === '' || !/[(),\[\]]/.test(input)) {
            return Atom(input);
        } else {
            log('warn', `Interpreting potentially complex string as Atom: "${input.substring(0, 50)}..."`);
            return Atom(input);
        }
    }

    export function unify(term1: Term, term2: Term, bindings: Map<string, Term> = new Map()): Map<string, Term> | null {
        const resolve = (term: Term, currentBindings: Map<string, Term>): Term => {
            let current = term;
            const visited = new Set<string>(); // Detect cycles during resolution
            while (current.kind === 'Variable' && currentBindings.has(current.name)) {
                if (visited.has(current.name)) return current; // Cycle detected, return var
                visited.add(current.name);
                current = currentBindings.get(current.name)!;
            }
            return current;
        };

        const occursCheck = (varName: string, termToCheck: Term, currentBindings: Map<string, Term>): boolean => {
            const resolved = resolve(termToCheck, currentBindings);
            if (resolved.kind === 'Variable') return varName === resolved.name;
            if (resolved.kind === 'Struct') return resolved.args.some(arg => occursCheck(varName, arg, currentBindings));
            if (resolved.kind === 'List') return resolved.elements.some(el => occursCheck(varName, el, currentBindings));
            return false;
        };

        const stack: [Term, Term][] = [[term1, term2]];
        let currentBindings = new Map(bindings);

        while (stack.length > 0) {
            const [t1, t2] = stack.pop()!;
            const rt1 = resolve(t1, currentBindings);
            const rt2 = resolve(t2, currentBindings);

            if (rt1 === rt2) continue; // Also handles resolved identical variables

            if (rt1.kind === 'Variable') {
                if (occursCheck(rt1.name, rt2, currentBindings)) return null;
                currentBindings.set(rt1.name, rt2);
                continue;
            }
            if (rt2.kind === 'Variable') {
                if (occursCheck(rt2.name, rt1, currentBindings)) return null;
                currentBindings.set(rt2.name, rt1);
                continue;
            }
            if (rt1.kind !== rt2.kind) return null;

            switch (rt1.kind) {
                case 'Atom': if (rt1.name !== (rt2 as Atom).name) return null; break;
                case 'Struct': {
                    const s2 = rt2 as Struct;
                    if (rt1.name !== s2.name || rt1.args.length !== s2.args.length) return null;
                    for (let i = rt1.args.length - 1; i >= 0; i--) stack.push([rt1.args[i], s2.args[i]]); // Push in reverse for stack order
                    break;
                }
                case 'List': {
                    const l2 = rt2 as List;
                    if (rt1.elements.length !== l2.elements.length) return null;
                    for (let i = rt1.elements.length - 1; i >= 0; i--) stack.push([rt1.elements[i], l2.elements[i]]); // Push in reverse
                    break;
                }
            }
        }
        return currentBindings;
    }

    export function substitute(term: Term, bindings: Map<string, Term>): Term {
        const resolve = (t: Term): Term => {
            let current = t;
            const visited = new Set<string>();
            while (current.kind === 'Variable' && bindings.has(current.name)) {
                if (visited.has(current.name)) return current; // Cycle detected
                visited.add(current.name);
                current = bindings.get(current.name)!;
            }
            return current;
        };

        const resolved = resolve(term);
        switch (resolved.kind) {
            case 'Struct': return { ...resolved, args: resolved.args.map(arg => substitute(arg, bindings)) };
            case 'List': return { ...resolved, elements: resolved.elements.map(el => substitute(el, bindings)) };
            default: return resolved; // Atom or unresolved Variable
        }
    }

    export function jsonToTerm(data: any): Term | null {
        if (data === null || typeof data === 'undefined') return Atom('null');
        if (typeof data === 'string') return Atom(data); // Try Atom first for strings
        if (typeof data === 'number' || typeof data === 'boolean') return Atom(String(data));

        if (Array.isArray(data)) {
            const elements = data.map(jsonToTerm).filter((t): t is Term => t !== null);
            return List(elements);
        }

        if (typeof data === 'object') {
            // Check if it matches our Term structure
            const { kind, name, args, elements } = data;
            switch (kind) {
                case 'Atom': return typeof name === 'string' ? Atom(name) : null;
                case 'Variable': return typeof name === 'string' ? Variable(name) : null;
                case 'Struct':
                    if (typeof name === 'string' && Array.isArray(args)) {
                        const convertedArgs = args.map(jsonToTerm).filter((t): t is Term => t !== null);
                        return convertedArgs.length === args.length ? Struct(name, convertedArgs) : null;
                    } return null;
                case 'List':
                    if (Array.isArray(elements)) {
                        const convertedElements = elements.map(jsonToTerm).filter((t): t is Term => t !== null);
                        return convertedElements.length === elements.length ? List(convertedElements) : null;
                    } return null;
            }
            // Fallback: Convert generic JSON object to Struct("json_object", [Struct(key, [value_term])])
            const objArgs = Object.entries(data)
                .map(([k, v]) => {
                    const termValue = jsonToTerm(v);
                    // Represent key-value pair as Struct(key, [value])
                    return termValue ? Struct(k, [termValue]) : null;
                })
                .filter((t): t is Struct => t !== null && t.kind === 'Struct');
            return Struct("json_object", objArgs);
        }
        log('warn', `Could not convert JSON data to Term: ${typeof data}`);
        return null; // Cannot convert
    }

    export function extractBindings(bindings: Map<string, Term>): Record<string, string> {
        const context: Record<string, string> = {};
        bindings.forEach((value, key) => { context[key] = toString(value); });
        return context;
    }
}

abstract class BaseStore<T extends { id: string, metadata?: { created?: string, modified?: string } }> {
    protected items = new Map<string, T>();
    private listeners: Set<() => void> = new Set();
    private changedItems: Set<string> = new Set();
    private deletedItems: Set<string> = new Set();
    private changeNotificationScheduled = false;

    add(item: T): void { this.update(item, true); }
    get(id: string): T | undefined { return this.items.get(id); }
    getAll(): T[] { return Array.from(this.items.values()); }
    count(): number { return this.items.size; }

    findItemByPrefix(prefix: string): T | undefined {
        if (!prefix || prefix.length < 3) return undefined;
        if (this.items.has(prefix)) return this.items.get(prefix); // Exact match first
        const matching = this.getAll().filter(item => item.id.startsWith(prefix));
        return matching.length === 1 ? matching[0] : undefined;
    }

    addChangeListener(listener: () => void): void { this.listeners.add(listener); }
    removeChangeListener(listener: () => void): void { this.listeners.delete(listener); }

    update(item: T, isNew: boolean = false): void {
        const existing = this.items.get(item.id);
        const now = new Date().toISOString();
        // Preserve created time if it exists, otherwise set it now
        const created = existing?.metadata?.created ?? item.metadata?.created ?? now;
        const newItem = {
            ...item,
            metadata: { ...(existing?.metadata), ...(item.metadata), created, modified: now }
        };
        this.items.set(item.id, newItem);
        this.changedItems.add(item.id);
        this.deletedItems.delete(item.id); // If it was deleted then re-added/updated
        this.scheduleNotifyChange();
    }

    delete(id: string): boolean {
        if (!this.items.has(id)) return false;
        this.items.delete(id);
        this.changedItems.delete(id); // Remove from changed if it was modified then deleted
        this.deletedItems.add(id);
        this.scheduleNotifyChange();
        return true;
    }

    // Provides delta and clears internal tracking
    getDelta(): { changed: T[], deleted: string[] } {
        const changed = Array.from(this.changedItems).map(id => this.items.get(id)).filter((item): item is T => !!item);
        const deleted = Array.from(this.deletedItems);
        this.changedItems.clear();
        this.deletedItems.clear();
        return { changed, deleted };
    }

    // Schedules notification to avoid rapid firing for batch updates
    private scheduleNotifyChange(): void {
        if (!this.changeNotificationScheduled) {
            this.changeNotificationScheduled = true;
            process.nextTick(() => {
                this.notifyChange();
                this.changeNotificationScheduled = false;
            });
        }
    }

    private notifyChange(): void { this.listeners.forEach(listener => listener()); }

    abstract toJSON(): Record<string, any>;
    abstract loadJSON(data: Record<string, any>): void;

    protected baseLoadJSON(data: Record<string, any>, itemFactory: (id: string, itemData: any) => T | null): void {
        this.items.clear();
        Object.entries(data).forEach(([id, itemData]) => {
            try {
                const item = itemFactory(id, itemData);
                if (item) this.items.set(id, item);
                else log('warn', `Skipped loading invalid item ${id} from JSON data.`);
            } catch (e: any) { log('error', `Failed to load item ${id}: ${e.message}`); }
        });
        this.changedItems.clear(); // Reset delta tracking after load
        this.deletedItems.clear();
    }
}

class Thoughts extends BaseStore<Thought> {
    getPending(): Thought[] { return this.getAll().filter(t => t.status === Status.PENDING); }
    findThought(idPrefix: string): Thought | undefined { return this.findItemByPrefix(idPrefix); }

    findPendingPrompt(promptId: string): Thought | undefined {
        return this.getAll().find(t => t.type === Type.USER_PROMPT && t.status === Status.PENDING && t.metadata.promptId === promptId);
    }

    findWaitingThought(waitingForId: string): Thought | undefined {
        return this.getAll().find(t => t.status === Status.WAITING && t.metadata.waitingFor === waitingForId);
    }

    findStuckThoughts(thresholdMs: number): Thought[] {
        const now = Date.now();
        return this.getAll().filter(t =>
            (t.status === Status.PENDING || t.status === Status.WAITING || t.status === Status.ACTIVE) && // Include ACTIVE
            t.metadata.modified &&
            (now - new Date(t.metadata.modified).getTime()) > thresholdMs
        );
    }

    getRootThoughts(): Thought[] { return this.getAll().filter(t => !t.metadata.rootId || t.metadata.rootId === t.id); }
    getDescendants(rootId: string): Thought[] { return this.getAll().filter(t => t.metadata.rootId === rootId && t.id !== rootId); }

    toJSON(): Record<string, any> {
        const obj: Record<string, any> = {};
        this.items.forEach((thought, id) => obj[id] = { ...thought, belief: thought.belief.toJSON() });
        return obj;
    }

    loadJSON(data: Record<string, any>): void {
        this.baseLoadJSON(data, (id, thoughtData) => {
            const { belief, content, metadata, status: loadedStatus, ...rest } = thoughtData;

            // Validate Status
            const status = Object.values(Status).includes(loadedStatus) ? loadedStatus : Status.PENDING;
            // Reset ACTIVE on load, they should become PENDING to be picked up again
            const finalStatus = status === Status.ACTIVE ? Status.PENDING : status;

            // Validate Content
            const contentTerm = Terms.jsonToTerm(content);
            if (!contentTerm) { log('error', `Failed to parse content for thought ${id}, skipping.`); return null; }

            // Ensure metadata exists
            const finalMetadata = metadata || {};

            // Ensure root tasks have a taskStatus, default to RUNNING
            if (!finalMetadata.rootId || finalMetadata.rootId === id) {
                finalMetadata.taskStatus = ['RUNNING', 'PAUSED'].includes(finalMetadata.taskStatus) ? finalMetadata.taskStatus : 'RUNNING';
            }

            return { ...rest, id, belief: Belief.fromJSON(belief), content: contentTerm, status: finalStatus, metadata: finalMetadata };
        });
        log('info', `Loaded ${this.count()} thoughts. Reset ACTIVE thoughts to PENDING.`);
    }
}

class Rules extends BaseStore<Rule> {
    findRule(idPrefix: string): Rule | undefined { return this.findItemByPrefix(idPrefix); }
    toJSON(): Record<string, any> {
        const obj: Record<string, any> = {};
        this.items.forEach((rule, id) => obj[id] = { ...rule, belief: rule.belief.toJSON() });
        return obj;
    }

    loadJSON(data: Record<string, any>): void {
        this.baseLoadJSON(data, (id, ruleData) => {
            const { belief, pattern, action, metadata, ...rest } = ruleData;
            const patternTerm = Terms.jsonToTerm(pattern);
            const actionTerm = Terms.jsonToTerm(action);
            if (!patternTerm || !actionTerm) { log('error', `Failed to parse pattern/action for rule ${id}, skipping.`); return null; }
            return { ...rest, id, belief: Belief.fromJSON(belief), pattern: patternTerm, action: actionTerm, metadata: metadata ?? {} };
        });
        log('info', `Loaded ${this.count()} rules.`);
    }
}

class Memory {
    private vectorStore: FaissStore | null = null;
    public isReady = false;
    constructor(private embeddings: Embeddings, private storePath: string) { }

    async initialize(): Promise<void> {
        if (this.isReady) return;
        try {
            // Ensure directory exists before trying to load/create
            await fs.mkdir(path.dirname(this.storePath), { recursive: true });
            try {
                // Try loading existing store
                this.vectorStore = await FaissStore.load(this.storePath, this.embeddings);
                this.isReady = true;
                log('info', `Vector store loaded from ${this.storePath}`);
            } catch (loadError: any) {
                // If loading fails (e.g., ENOENT or corrupted), create a new one
                if (loadError.code === 'ENOENT' || loadError.message?.includes('No such file or directory')) {
                    log('warn', `Vector store not found at ${this.storePath}, initializing new store.`);
                } else {
                    log('warn', `Failed to load vector store (error: ${loadError.message}). Initializing new store.`);
                }
                // Create a minimal store
                const dummyDoc = new Document({
                    pageContent: "Initial sentinel document",
                    metadata: { sourceId: "init", type: "SYSTEM", created: new Date().toISOString(), id: "init_doc", _docId: generateId() }
                });
                this.vectorStore = await FaissStore.fromDocuments([dummyDoc], this.embeddings);
                await this.save(); // Save the new empty store immediately
                this.isReady = true;
                log('info', `New vector store created at ${this.storePath}`);
            }
        } catch (initError: any) {
            log('error', 'Failed to initialize vector store directory or create new store:', initError.message);
            this.isReady = false; // Ensure readiness is false on failure
        }
    }

    async add(entry: Omit<MemoryEntry, 'embedding' | 'score'>): Promise<void> {
        if (!this.vectorStore || !this.isReady) { log('warn', 'MemoryStore not ready, cannot add entry.'); return; }
        // Ensure unique document ID for FAISS internal tracking
        const docMetadata = { ...entry.metadata, id: entry.id, _docId: generateId() };
        const doc = new Document({ pageContent: entry.content, metadata: docMetadata });
        try {
            await this.vectorStore.addDocuments([doc]);
            // Debounce saving? For now, save on add. Consider performance implications.
            await this.save();
            log('debug', `Added memory entry from source ${entry.metadata.sourceId}`);
        } catch (error: any) { log('error', 'Failed to add document to vector store:', error.message); }
    }

    async search(query: string, k: number = 5): Promise<MemoryEntry[]> {
        if (!this.vectorStore || !this.isReady) { log('warn', 'MemoryStore not ready, cannot search.'); return []; }
        try {
            const results = await this.vectorStore.similaritySearchWithScore(query, k);
            log('debug', `Memory search for "${query.substring(0, 30)}..." returned ${results.length} results.`);
            return results
                .filter(([doc]) => doc.metadata.sourceId !== 'init') // Exclude the sentinel doc
                .map(([doc, score]): MemoryEntry => ({
                    // Use the 'id' we stored, not LangChain's internal doc ID (_docId)
                    id: doc.metadata.id ?? generateId(), // Fallback ID if missing
                    content: doc.pageContent,
                    metadata: {
                        created: doc.metadata.created ?? new Date().toISOString(),
                        type: doc.metadata.type ?? 'UNKNOWN',
                        sourceId: doc.metadata.sourceId ?? 'UNKNOWN'
                    },
                    score: score,
                    embedding: [] // Embedding not typically returned by search
                }));
        } catch (error: any) { log('error', 'Failed to search vector store:', error.message); return []; }
    }

    async save(): Promise<void> {
        if (this.vectorStore && this.isReady) {
            try {
                await this.vectorStore.save(this.storePath);
            } catch (error: any) {
                log('error', 'Failed to save vector store:', error.message);
            }
        }
    }
}

class LLM {
    llm: BaseChatModel;
    embeddings: Embeddings;
    constructor() {
        this.llm = new ChatOllama({ baseUrl: OLLAMA_BASE_URL, model: OLLAMA_MODEL, temperature: 0.7 });
        this.embeddings = new OllamaEmbeddings({ model: OLLAMA_EMBEDDING_MODEL, baseUrl: OLLAMA_BASE_URL });
        log('info', `LLM Service: Model=${OLLAMA_MODEL}, Embeddings=${OLLAMA_EMBEDDING_MODEL}, URL=${OLLAMA_BASE_URL}`);
    }

    async generate(prompt: string): Promise<string> {
        log('debug', `LLM Prompt: "${prompt.substring(0, 100)}..."`);
        try {
            // Use invoke directly for simpler cases, pipe if chaining needed
            const response = await this.llm.invoke([new HumanMessage(prompt)]);
            const textResponse = typeof response.content === 'string' ? response.content : JSON.stringify(response.content);
            const trimmedResponse = textResponse.trim();
            log('debug', `LLM Response: "${trimmedResponse.substring(0, 100)}..."`);
            return trimmedResponse;
        } catch (error: any) {
            log('error', `LLM generation failed: ${error.message}`);
            // Don't return the error message itself as a valid response
            throw new Error(`LLM generation failed: ${error.message}`);
        }
    }

    async embed(text: string): Promise<number[]> {
        log('debug', `Embedding text: "${text.substring(0, 50)}..."`);
        try {
            return await this.embeddings.embedQuery(text);
        } catch (error: any) {
            log('error', `Embedding failed: ${error.message}`);
            throw new Error(`Embedding failed: ${error.message}`); // Propagate error
        }
    }
}

type PromptKey =
    'GENERATE_GOAL' | 'GENERATE_STRATEGY' | 'GENERATE_OUTCOME' | 'SUGGEST_GOAL' |
    'FALLBACK_ASK_USER' | 'CONFIRM_GOAL' | 'CLARIFY_INPUT' | 'CONFIRM_STRATEGY' |
    'NEXT_TASK_PROMPT' | 'HANDLE_FAILURE' | 'GENERATE_RECOVERY_STRATEGY' |
    'STUCK_THOUGHT_PROMPT' | 'SUMMARIZE_TASK_COMPLETION' | 'ASK_FOR_RULE_SUGGESTION';

class Prompts {
    private prompts: Record<PromptKey, string> = {
        GENERATE_GOAL: 'Input: "{input}". Define a specific, actionable GOAL based on this input. Output ONLY the goal text.',
        GENERATE_STRATEGY: 'Goal: "{goal}". Outline 1-3 concrete STRATEGY steps to achieve this goal. Output each step on a new line, starting with "- ".',
        GENERATE_OUTCOME: 'Strategy step "{strategy}" was attempted. Describe a likely concise OUTCOME. Output ONLY the outcome text.',
        SUGGEST_GOAL: 'Based on the current context "{context}"{memoryContext}\n\nSuggest ONE concise, actionable next goal or task. Output only the suggested goal text.',
        FALLBACK_ASK_USER: 'No rule found for thought {thoughtId} ({thoughtType}: "{thoughtContent}"). How should I proceed with this?',
        CONFIRM_GOAL: 'Goal generated: "{goal}". Is this correct, or should it be refined? Please respond.',
        CLARIFY_INPUT: 'The input "{input}" seems brief or ambiguous. Could you please provide more details or context for a better goal?',
        CONFIRM_STRATEGY: 'Proposed strategy step: "{strategy}". Does this seem like a good step towards the goal? (yes/no/suggest alternative)',
        NEXT_TASK_PROMPT: 'Task "{taskContent}" ({taskId}) is complete. What would you like to work on next?',
        HANDLE_FAILURE: 'Thought {failedThoughtId} ("{content}") failed with error: "{error}". How should I proceed? (Options: retry / abandon / edit_goal / provide_new_strategy / manual_fix)',
        GENERATE_RECOVERY_STRATEGY: "Thought {failedId} ('{failedContent}') failed with error: '{error}'. Suggest a concise alternative strategy or action to achieve the original goal (if discernible). Output ONLY the suggested action/strategy text.",
        STUCK_THOUGHT_PROMPT: "Thought {thoughtId} has been in status '{status}' for {age}. What should be done? (Options: retry / mark_failed / investigate / ignore)",
        SUMMARIZE_TASK_COMPLETION: "Task goal '{goalContent}' (ID: {goalId}) is complete. Based on its associated thoughts (Outcomes, Facts, Logs), provide a brief summary of what was accomplished and any key findings.",
        ASK_FOR_RULE_SUGGESTION: "Processing pattern '{pattern}' led to failure '{error}' multiple times. Can you suggest a better rule (pattern -> action) to handle this situation more effectively?",
    };

    format(key: PromptKey, context: Record<string, string>): string {
        let template = this.prompts[key];
        if (!template) {
            log('error', `Prompt template key "${key}" not found.`);
            return `Error: Prompt template "${key}" not found. Context: ${JSON.stringify(context)}`;
        }
        // Replace placeholders like {key}
        Object.entries(context).forEach(([k, v]) => {
            template = template.replace(new RegExp(`\\{${k}\\}`, 'g'), String(v ?? '')); // Ensure v is string
        });
        // Remove any remaining unused placeholders like {unused_key} to avoid confusing the LLM
        template = template.replace(/\{[a-zA-Z0-9_]+\}/g, '');
        return template;
    }
}

// --- Tools ---
abstract class BaseTool implements Tool {
    abstract name: string;
    abstract description: string;
    abstract execute(actionTerm: Struct, context: ToolContext, trigger: Thought): Promise<Term | null>;

    protected createResultAtom(success: boolean, code: string, message?: string): Atom {
        const prefix = success ? 'ok' : 'error';
        // Sanitize message for atom compatibility (replace spaces, limit length)
        const safeMessage = message ? `:${message.replace(/\s+/g, '_').substring(0, 50)}` : '';
        const atomName = `${prefix}:${this.name}_${code}${safeMessage}`;
        return Atom(atomName);
    }

    protected getRuleBindings(trigger: Thought, ruleContext: ToolContext): Map<string, Term> | null {
        const ruleId = trigger.metadata.ruleId;
        if (!ruleId) return null;
        const rule = ruleContext.rules.get(ruleId);
        if (!rule) return null;
        // Re-run unification to get bindings specific to this trigger/rule pair
        return Terms.unify(rule.pattern, trigger.content);
    }

    // Extracts context values, resolving variables using rule bindings if possible
    protected extractContextArgs(contextTerm: Term | undefined, trigger: Thought, ruleContext: ToolContext): Record<string, string> {
        const promptContext: Record<string, string> = {};
        if (contextTerm?.kind !== 'Struct' || contextTerm.name !== 'context') return promptContext;

        const bindings = this.getRuleBindings(trigger, ruleContext); // Get bindings for variable resolution

        contextTerm.args.forEach(arg => {
            let key: string | null = null;
            let value: string | null = null;

            // Format 1: Atom("key:value") or Atom("key:?Var")
            if (arg.kind === 'Atom' && arg.name.includes(':')) {
                const [k, ...vParts] = arg.name.split(':');
                key = k;
                value = vParts.join(':');
                if (value.startsWith('?') && bindings) {
                    const varName = value.substring(1);
                    const boundValue = bindings.get(varName);
                    if (boundValue) value = Terms.toString(boundValue);
                }
            }
            // Format 2: Struct(key, [Variable("Var")]) or Struct(key, [Atom("value")])
            else if (arg.kind === 'Struct' && arg.args.length === 1) {
                key = arg.name;
                const valueTerm = arg.args[0];
                if (valueTerm.kind === 'Variable' && bindings) {
                    const boundValue = bindings.get(valueTerm.name);
                    if (boundValue) value = Terms.toString(boundValue);
                    else value = `?${valueTerm.name}`; // Variable not bound
                } else {
                    value = Terms.toString(valueTerm);
                }
            }

            if (key && value !== null) {
                promptContext[key] = value;
            } else {
                log('warn', `Could not parse context argument: ${Terms.format(arg, false)} in ${this.name}`);
            }
        });
        return promptContext;
    }

    // Helper to resolve target ID from term (Atom("id_prefix") or Atom("self"))
    protected resolveTargetId(t: Term | undefined, fallbackId: string, context: ToolContext): string | null {
        if (t?.kind !== 'Atom') return null;
        if (t.name === 'self') return fallbackId;
        // Try finding thought first, then rule
        const target = context.thoughts.findThought(t.name) ?? context.rules.findRule(t.name);
        // Allow full UUIDs even if not found locally (might be targeting something external/future)
        return target?.id ?? (isValidUuid(t.name) ? t.name : null);
    }
}

class LLMTool extends BaseTool {
    name = "LLMTool";
    description = "Interacts with the LLM: generate(prompt_key_atom, context_term), embed(text_term).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        switch (operation) {
            case 'generate': return this.generate(action, context, trigger);
            case 'embed': return this.embed(action, context);
            default: return this.createResultAtom(false, "unsupported_operation", operation ?? 'missing');
        }
    }

    private async embed(action: Struct, context: ToolContext): Promise<Term> {
        const inputTerm = action.args[1];
        if (!inputTerm) return this.createResultAtom(false, "invalid_params", "Missing text term for embed");
        try {
            const inputText = Terms.toString(inputTerm);
            await context.llm.embed(inputText); // We don't typically return the embedding itself
            return this.createResultAtom(true, "embedded");
        } catch (error: any) {
            return this.createResultAtom(false, "embedding_failed", error.message);
        }
    }

    private async generate(action: Struct, context: ToolContext, trigger: Thought): Promise<Term> {
        const promptKeyAtom = action.args[1];
        const contextTerm = action.args[2]; // Optional context struct: context(key1:val1, key2:?Var2)

        if (promptKeyAtom?.kind !== 'Atom') return this.createResultAtom(false, "invalid_params", "Missing prompt key atom");
        if (contextTerm && (contextTerm.kind !== 'Struct' || contextTerm.name !== 'context')) {
            return this.createResultAtom(false, "invalid_params", "Context must be a Struct named 'context'");
        }

        const promptKey = promptKeyAtom.name as PromptKey;
        const promptContext = this.extractContextArgs(contextTerm, trigger, context);
        const prompt = context.prompts.format(promptKey, promptContext);

        try {
            const response = await context.llm.generate(prompt);
            // Attempt to parse LLM response as a Term (e.g., if it returns `Struct(...)`)
            try {
                const parsedTerm = Terms.fromString(response);
                if (parsedTerm) return parsedTerm;
            } catch { /* Ignore parsing error, treat as Atom */ }
            // Fallback: return raw response as Atom
            return Atom(response);
        } catch (error: any) {
            return this.createResultAtom(false, "generation_failed", error.message);
        }
    }
}

class MemoryTool extends BaseTool {
    name = "MemoryTool";
    description = "Manages vector memory: add(content_term), search(query_term, k?).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (!operation) return this.createResultAtom(false, "missing_operation");

        switch (operation) {
            case 'add': {
                const contentTerm = action.args[1];
                if (!contentTerm) return this.createResultAtom(false, "missing_add_content");
                try {
                    await context.memory.add({
                        id: generateId(), // Generate unique ID for memory entry itself
                        content: Terms.toString(contentTerm),
                        metadata: {
                            created: new Date().toISOString(),
                            type: trigger.type, // Type of the thought triggering the add
                            sourceId: trigger.id // ID of the thought triggering the add
                        }
                    });
                    // Mark the original thought as embedded (if desired)
                    trigger.metadata.embedded = new Date().toISOString();
                    context.thoughts.update(trigger); // Update the trigger thought
                    return this.createResultAtom(true, "memory_added", shortId(trigger.id));
                } catch (error: any) {
                    return this.createResultAtom(false, "add_failed", error.message);
                }
            }
            case 'search': {
                const queryTerm = action.args[1];
                const kTerm = action.args[2];
                if (!queryTerm) return this.createResultAtom(false, "missing_search_query");
                const queryStr = Terms.toString(queryTerm);
                // Default k=3 if not specified or invalid
                const k = (kTerm?.kind === 'Atom' && /^\d+$/.test(kTerm.name)) ? parseInt(kTerm.name, 10) : 3;
                try {
                    const results = await context.memory.search(queryStr, k);
                    // Return results as a List of Atoms containing the content
                    return List(results.map(r => Atom(r.content)));
                } catch (error: any) {
                    return this.createResultAtom(false, "search_failed", error.message);
                }
            }
            default: return this.createResultAtom(false, "unsupported_operation", operation);
        }
    }
}

class UserInteractionTool extends BaseTool {
    name = "UserInteractionTool";
    description = "Requests input from the user: prompt(prompt_text_term | prompt_key_atom, context_term?).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (operation !== 'prompt') return this.createResultAtom(false, "invalid_operation", operation ?? 'missing');

        const promptSource = action.args[1];
        const contextTerm = action.args[2]; // Optional context struct for prompt formatting
        if (!promptSource) return this.createResultAtom(false, "invalid_params", "Missing prompt source (text or key)");

        let promptText: string;
        try {
            if (promptSource.kind === 'Atom') {
                const promptKey = promptSource.name as PromptKey;
                const promptContext = this.extractContextArgs(contextTerm, trigger, context);
                promptText = context.prompts.format(promptKey, promptContext);
            } else {
                // If source is not an Atom, treat it as the literal prompt text
                promptText = Terms.toString(promptSource);
            }
        } catch (error: any) {
            return this.createResultAtom(false, "prompt_format_error", error.message);
        }

        const promptThoughtId = generateId();
        // Create the USER_PROMPT thought
        context.engine.addThought({
            id: promptThoughtId,
            type: Type.USER_PROMPT,
            content: Atom(promptText),
            belief: Belief.DEFAULT,
            status: Status.PENDING, // It's pending until the user sees and responds
            metadata: {
                rootId: trigger.metadata.rootId ?? trigger.id,
                parentId: trigger.id,
                created: new Date().toISOString(),
                promptId: promptThoughtId, // Link back to itself for response matching
                provenance: this.name
            }
        });

        // Set the triggering thought to WAITING
        trigger.status = Status.WAITING;
        trigger.metadata.waitingFor = promptThoughtId; // Wait specifically for this prompt ID
        context.thoughts.update(trigger);

        log('info', `User prompt requested (${shortId(promptThoughtId)}) for thought ${shortId(trigger.id)}.`);
        return this.createResultAtom(true, "prompt_requested", promptThoughtId);
    }
}

class GoalProposalTool extends BaseTool {
    name = "GoalProposalTool";
    description = "Suggests new goals based on context: suggest(context_term?).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (operation !== 'suggest') return this.createResultAtom(false, "invalid_operation", operation ?? 'missing');

        // Use provided context term if available, otherwise use trigger's content
        const contextTerm = action.args[1] ?? trigger.content;
        const contextStr = Terms.toString(contextTerm);

        try {
            // Enhance context with related memory
            const memoryResults = await context.memory.search(`Relevant past goals or outcomes related to: ${contextStr}`, 3);
            const memoryContext = memoryResults.length > 0
                ? `\nRelated past activities:\n - ${memoryResults.map(r => r.content).join("\n - ")}`
                : "";

            const prompt = context.prompts.format('SUGGEST_GOAL', { context: contextStr, memoryContext: memoryContext });
            const suggestionText = await context.llm.generate(prompt);

            // Create a new GOAL thought with the suggestion
            const suggestionThought: Thought = {
                id: generateId(),
                type: Type.GOAL,
                content: Atom(suggestionText),
                belief: new Belief(1, 0), // Start with neutral-positive belief
                status: Status.PENDING,
                metadata: {
                    rootId: trigger.metadata.rootId ?? trigger.id, // Associate with the same task/root
                    parentId: trigger.id, // Triggering thought is the parent
                    created: new Date().toISOString(),
                    tags: ['suggested_goal'],
                    provenance: this.name
                }
            };
            context.engine.addThought(suggestionThought);
            log('info', `Suggested goal ${shortId(suggestionThought.id)} based on ${shortId(trigger.id)}.`);
            return this.createResultAtom(true, "suggestion_created", shortId(suggestionThought.id));

        } catch (error: any) {
            log('error', `${this.name} failed: ${error.message}`);
            return this.createResultAtom(false, "suggestion_failed", error.message);
        }
    }
}

class CoreTool extends BaseTool {
    name = "CoreTool";
    description = "Manages internal state: set_status(target_id, status), add_thought(type, content, root?, parent?), delete_thought(target_id), set_content(target_id, content), log(level, message_term).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (!operation) return this.createResultAtom(false, "missing_operation");

        const triggerId = trigger.id;
        switch (operation) {
            case 'set_status': return this.setStatus(action, triggerId, context, trigger);
            case 'add_thought': return this.addThought(action, triggerId, context, trigger);
            case 'delete_thought': return this.delThought(action, triggerId, context);
            case 'set_content': return this.setContent(action, triggerId, context);
            case 'log': return this.logMessage(action); // Context not needed for simple logging
            default: return this.createResultAtom(false, "unsupported_operation", operation);
        }
    }

    private setContent(action: Struct, triggerId: string, context: ToolContext): Term {
        const targetId = this.resolveTargetId(action.args[1], triggerId, context);
        const newContentTerm = action.args[2];
        if (!targetId || !newContentTerm) return this.createResultAtom(false, "invalid_set_content_params");

        const targetThought = context.thoughts.get(targetId);
        if (!targetThought) return this.createResultAtom(false, "target_not_found", shortId(targetId));

        targetThought.content = newContentTerm;
        context.thoughts.update(targetThought);
        log('debug', `Set content of ${shortId(targetThought.id)} via ${shortId(triggerId)}`);
        return this.createResultAtom(true, "content_set", shortId(targetThought.id));
    }

    private delThought(action: Struct, triggerId: string, context: ToolContext): Term {
        const targetId = this.resolveTargetId(action.args[1], triggerId, context);
        if (!targetId) return this.createResultAtom(false, "invalid_delete_thought_params");

        const deleted = context.thoughts.delete(targetId);
        log('debug', `Attempted delete thought ${shortId(targetId)} via ${shortId(triggerId)}: ${deleted}`);
        // TODO: Consider deleting descendants? For now, just the target.
        return this.createResultAtom(deleted, deleted ? "thought_deleted" : "delete_failed", shortId(targetId));
    }

    private addThought(action: Struct, triggerId: string, context: ToolContext, trigger: Thought): Term {
        const typeTerm = action.args[1];
        const contentTerm = action.args[2];
        const rootIdTerm = action.args[3]; // Optional: Atom("id") or Atom("self")
        const parentIdTerm = action.args[4]; // Optional: Atom("id") or Atom("self")

        if (typeTerm?.kind !== 'Atom' || !contentTerm) return this.createResultAtom(false, "invalid_add_thought_params");
        const type = typeTerm.name.toUpperCase() as Type;
        if (!Object.values(Type).includes(type)) return this.createResultAtom(false, "invalid_thought_type", typeTerm.name);

        // Determine parent: use specified term, fallback to 'self' (trigger), then triggerId
        const parentId = this.resolveTargetId(parentIdTerm, triggerId, context) ?? triggerId;
        const parentThought = context.thoughts.get(parentId);

        // Determine root: use specified term, fallback to parent's root, then parentId itself
        const defaultRootId = parentThought?.metadata.rootId ?? parentId;
        const rootId = this.resolveTargetId(rootIdTerm, defaultRootId, context) ?? defaultRootId;

        const newThought: Thought = {
            id: generateId(), type, content: contentTerm, belief: Belief.DEFAULT, status: Status.PENDING,
            metadata: {
                rootId, parentId, created: new Date().toISOString(),
                provenance: `${this.name} (rule ${shortId(trigger.metadata.ruleId)} on ${shortId(triggerId)})`
            }
        };
        // If this new thought is its own root, mark it as RUNNING by default
        if (newThought.id === newThought.metadata.rootId) newThought.metadata.taskStatus = 'RUNNING';

        context.engine.addThought(newThought);
        log('debug', `Added thought ${shortId(newThought.id)} (${type}) via ${shortId(triggerId)}`);
        return this.createResultAtom(true, "thought_added", shortId(newThought.id));
    }

    private setStatus(action: Struct, triggerId: string, context: ToolContext, trigger: Thought): Term {
        const targetId = this.resolveTargetId(action.args[1], triggerId, context);
        const newStatusTerm = action.args[2];
        if (!targetId || newStatusTerm?.kind !== 'Atom') return this.createResultAtom(false, "invalid_set_status_params");

        const newStatus = newStatusTerm.name.toUpperCase() as Status;
        if (!Object.values(Status).includes(newStatus)) return this.createResultAtom(false, "invalid_status_value", newStatusTerm.name);

        const targetThought = context.thoughts.get(targetId);
        if (!targetThought) return this.createResultAtom(false, "target_not_found", shortId(targetId));

        const oldStatus = targetThought.status;
        if (oldStatus === newStatus) {
            log('debug', `Status of ${shortId(targetThought.id)} already ${newStatus}. No change.`);
            return this.createResultAtom(true, "status_unchanged", `${shortId(targetThought.id)}_${newStatus}`);
        }

        targetThought.status = newStatus;
        // Clear waiting state if moving out of WAITING
        if (oldStatus === Status.WAITING && newStatus !== Status.WAITING) {
            delete targetThought.metadata.waitingFor;
        }
        // Clear error/retries if moving to a non-failure/non-pending state
        if (newStatus !== Status.FAILED && newStatus !== Status.PENDING) {
            delete targetThought.metadata.error;
            delete targetThought.metadata.retries;
        }

        context.thoughts.update(targetThought);
        log('debug', `Set status of ${shortId(targetThought.id)} to ${newStatus} via ${shortId(triggerId)}`);

        // Check for task completion (root GOAL moving to DONE)
        if (oldStatus !== Status.DONE && newStatus === Status.DONE && targetThought.type === Type.GOAL && targetThought.metadata.rootId === targetThought.id) {
            log('info', `Task ${shortId(targetThought.id)} marked as DONE.`);
            // Add a log event to potentially trigger summarization or next steps
            context.engine.addThought({
                id: generateId(), type: Type.LOG, content: Struct("task_completed", [Atom(targetThought.id)]),
                belief: Belief.DEFAULT, status: Status.PENDING,
                metadata: {
                    rootId: targetThought.id, parentId: targetThought.id, created: new Date().toISOString(),
                    provenance: `${this.name}_status_change`
                }
            });
        }

        return this.createResultAtom(true, "status_set", `${shortId(targetThought.id)}_to_${newStatus}`);
    }

    private logMessage(action: Struct): Term {
        const levelTerm = action.args[1];
        const messageTerm = action.args[2];
        if (levelTerm?.kind !== 'Atom' || !messageTerm) return this.createResultAtom(false, "invalid_log_params");

        const level = levelTerm.name.toLowerCase() as LogLevel;
        if (!Object.values(LOG_LEVELS).includes(LOG_LEVELS[level])) return this.createResultAtom(false, "invalid_log_level", level);

        const message = Terms.toString(messageTerm);
        log(level, `[CoreTool Log] ${message}`);
        // Logging itself is usually successful unless params are wrong
        return this.createResultAtom(true, "logged", level);
    }
}

class TimeTool extends BaseTool {
    name = "TimeTool";
    description = "Provides time functions: get_time(), wait(duration_ms), schedule(action_term, time_ms), periodic_tick(interval_ms).";
    private static tickIntervalId: NodeJS.Timeout | null = null; // Static to ensure only one global tick interval

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (!operation) return this.createResultAtom(false, "missing_operation");

        switch (operation) {
            case 'get_time': return Atom(Date.now().toString());
            case 'wait': {
                const durationTerm = action.args[1];
                if (durationTerm?.kind !== 'Atom' || !/^\d+$/.test(durationTerm.name)) return this.createResultAtom(false, "invalid_wait_duration");
                const durationMs = parseInt(durationTerm.name, 10);
                if (durationMs <= 0) return this.createResultAtom(false, "invalid_wait_duration", "Duration must be positive");

                const wakeUpTime = Date.now() + durationMs;
                trigger.status = Status.WAITING;
                trigger.metadata.waitingFor = { type: 'time', timestamp: wakeUpTime };
                context.thoughts.update(trigger);
                log('debug', `Thought ${shortId(trigger.id)} waiting for ${durationMs}ms until ${new Date(wakeUpTime).toISOString()}`);
                return this.createResultAtom(true, "wait_set", durationTerm.name);
            }
            case 'schedule': { // TODO: Implement scheduling mechanism
                log('warn', `TimeTool 'schedule' operation is not yet implemented.`);
                return this.createResultAtom(false, "unsupported_operation", "schedule");
            }
            case 'periodic_tick': {
                // This sets up the *engine* to generate tick thoughts.
                const intervalTerm = action.args[1];
                if (intervalTerm?.kind !== 'Atom' || !/^\d+$/.test(intervalTerm.name)) return this.createResultAtom(false, "invalid_tick_interval");
                const intervalMs = parseInt(intervalTerm.name, 10);
                if (intervalMs <= 100) return this.createResultAtom(false, "invalid_tick_interval", "Interval too short");

                // Clear existing interval if setting a new one
                if (TimeTool.tickIntervalId) clearInterval(TimeTool.tickIntervalId);

                TimeTool.tickIntervalId = setInterval(() => {
                    // Add a SYSTEM thought of type 'tick' for rules to match on
                    context.engine.addThought({
                        id: generateId(), type: Type.SYSTEM, content: Struct("tick", [Atom(Date.now().toString())]),
                        belief: Belief.DEFAULT, status: Status.PENDING,
                        metadata: { created: new Date().toISOString(), provenance: this.name, priority: -100 } // Low priority system event
                    });
                }, intervalMs);

                log('info', `Periodic tick event generation started every ${intervalMs}ms`);
                // The action itself succeeded in setting up the interval
                return this.createResultAtom(true, "tick_started", intervalTerm.name);
            }
            default: return this.createResultAtom(false, "unsupported_operation", operation);
        }
    }
}

class FileSystemTool extends BaseTool {
    name = "FileSystemTool";
    description = "Interacts with local filesystem (Restricted to DATA_DIR): read_file(path_atom), write_file(path_atom, content_term), list_dir(path_atom).";

    private resolveAndCheckPath(relativePath: string): string {
        // Resolve relative to DATA_DIR
        const absolutePath = path.resolve(DATA_DIR, relativePath);
        // Security Check: Ensure the resolved path is still within DATA_DIR
        if (!absolutePath.startsWith(path.resolve(DATA_DIR))) {
            throw new Error(`Security violation: Path '${relativePath}' resolves outside of designated data directory.`);
        }
        return absolutePath;
    }

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        const pathTerm = action.args[1];
        if (!operation || pathTerm?.kind !== 'Atom') return this.createResultAtom(false, "invalid_params", "Operation or path missing/invalid");

        try {
            const filePath = this.resolveAndCheckPath(pathTerm.name);
            const dirPath = path.dirname(filePath); // Needed for write/list

            switch (operation) {
                case 'read_file':
                    const content = await fs.readFile(filePath, 'utf-8');
                    // Return content as a single Atom
                    return Atom(content);
                case 'write_file':
                    const contentTerm = action.args[2];
                    if (!contentTerm) return this.createResultAtom(false, "missing_write_content");
                    // Ensure directory exists before writing
                    await fs.mkdir(dirPath, { recursive: true });
                    await fs.writeFile(filePath, Terms.toString(contentTerm), 'utf-8');
                    log('info', `Wrote file: ${pathTerm.name} via ${shortId(trigger.id)}`);
                    return this.createResultAtom(true, "file_written", pathTerm.name);
                case 'list_dir':
                    // Ensure directory exists before listing
                    await fs.mkdir(filePath, { recursive: true }); // Use filePath as dir for list_dir
                    const files = await fs.readdir(filePath);
                    // Return file/dir names as a List of Atoms
                    return List(files.map(f => Atom(f)));
                default: return this.createResultAtom(false, "unsupported_operation", operation);
            }
        } catch (error: any) {
            log('error', `FileSystemTool error (${operation} ${pathTerm.name}): ${error.message}`);
            // Provide a more specific error code if available (e.g., ENOENT)
            return this.createResultAtom(false, "fs_error", error.code ?? error.message);
        }
    }
}

class WebSearchTool extends BaseTool {
    name = "WebSearchTool";
    description = "Performs web searches: search(query_term, num_results?). (Requires external setup/API key - NOT IMPLEMENTED)";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        log('warn', "WebSearchTool is not implemented. Requires external API integration (e.g., SerpApi, Tavily).");
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (operation !== 'search') return this.createResultAtom(false, "unsupported_operation", operation ?? 'missing');

        const queryTerm = action.args[1];
        // const kTerm = action.args[2]; // Optional num results
        if (!queryTerm) return this.createResultAtom(false, "missing_search_query");
        const queryStr = Terms.toString(queryTerm);

        // Placeholder for actual implementation
        // try {
        //   const results = await someSearchApi(queryStr, k);
        //   return List(results.map(r => Struct("search_result", [Atom(r.title), Atom(r.link), Atom(r.snippet)])));
        // } catch (error: any) {
        //    return this.createResultAtom(false, "search_api_failed", error.message);
        // }

        return this.createResultAtom(false, "not_implemented", `Search for '${queryStr.substring(0, 30)}...'`);
    }
}

class ContextSummarizer extends BaseTool {
    name = "ContextSummarizer";
    description = "Summarizes task context: summarize(rootId_atom).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (operation !== 'summarize') return this.createResultAtom(false, "unsupported_operation", operation ?? 'missing');

        const rootIdTerm = action.args[1];
        // const maxTokensTerm = action.args[2]; // Optional token limit - ignore for now

        if (rootIdTerm?.kind !== 'Atom') return this.createResultAtom(false, "missing_root_id");
        // Resolve root ID using the standard helper
        const rootId = this.resolveTargetId(rootIdTerm, trigger.metadata.rootId ?? trigger.id, context);
        if (!rootId) return this.createResultAtom(false, "root_id_not_found", rootIdTerm.name);

        const rootThought = context.thoughts.get(rootId);
        if (!rootThought) return this.createResultAtom(false, "root_thought_not_found", shortId(rootId));

        const descendants = context.thoughts.getDescendants(rootId);
        // Include relevant types for summary
        const relevantThoughts = [rootThought, ...descendants].filter(t =>
            [Type.GOAL, Type.STRATEGY, Type.OUTCOME, Type.FACT, Type.LOG].includes(t.type)
        );

        if (relevantThoughts.length === 0) {
            return Atom("No relevant context found for summarization.");
        }

        // Sort by creation time for chronological context
        const contextText = relevantThoughts
            .sort((a, b) => (a.metadata.created ?? '').localeCompare(b.metadata.created ?? ''))
            .map(t => `${t.type}: ${Terms.toString(t.content)}`)
            .join('\n');

        // Limit context size sent to LLM
        const maxContextLength = 2000;
        const truncatedContext = contextText.length > maxContextLength
            ? contextText.substring(contextText.length - maxContextLength) // Take the most recent context
            : contextText;

        const prompt = context.prompts.format('SUMMARIZE_TASK_COMPLETION', {
            goalContent: Terms.toString(rootThought.content),
            goalId: shortId(rootThought.id)
        }) + `\n\nContext:\n---\n${truncatedContext}\n---\nSummary:`;

        try {
            const summary = await context.llm.generate(prompt);
            // Return summary as Atom or a structured term if needed later
            // context.engine.addThought({ type: Type.CONTEXT_SUMMARY ... }) ? Handled by rules instead.
            return Atom(summary);
        } catch (error: any) {
            log('error', `${this.name} LLM summarization failed: ${error.message}`);
            return this.createResultAtom(false, "llm_failed", error.message);
        }
    }
}

class RuleManager extends BaseTool {
    name = "RuleManager";
    description = "Manages rules (Use with extreme caution!): add_rule(pattern_term, action_term, priority?, desc?), delete_rule(ruleId_atom), modify_priority(ruleId_atom, delta_num).";

    async execute(action: Struct, context: ToolContext, trigger: Thought): Promise<Term | null> {
        log('warn', `Executing RuleManager operation via thought ${shortId(trigger.id)}`);
        const operation = action.args[0]?.kind === 'Atom' ? action.args[0].name : null;
        if (!operation) return this.createResultAtom(false, "missing_operation");

        try {
            switch (operation) {
                case 'add_rule': return this.addRule(action, context, trigger);
                case 'delete_rule': return this.deleteRule(action, context);
                case 'modify_priority': return this.modifyPriority(action, context);
                default: return this.createResultAtom(false, "unsupported_operation", operation);
            }
        } catch (error: any) {
            log('error', `RuleManager error (${operation}): ${error.message}`);
            return this.createResultAtom(false, "rule_mgmt_error", error.message);
        }
    }

    private addRule(action: Struct, context: ToolContext, trigger: Thought): Term {
        const patternTerm = action.args[1];
        const actionTerm = action.args[2];
        const priorityTerm = action.args[3];
        const descTerm = action.args[4];

        if (!patternTerm || !actionTerm) return this.createResultAtom(false, "invalid_add_rule_params");
        // Default priority 0, allow negative
        const priority = (priorityTerm?.kind === 'Atom' && /^-?\d+$/.test(priorityTerm.name)) ? parseInt(priorityTerm.name, 10) : 0;
        const description = descTerm ? Terms.toString(descTerm) : `Added by ${shortId(trigger.id)}`;

        const newRule: Rule = {
            id: generateId(), pattern: patternTerm, action: actionTerm, belief: Belief.DEFAULT,
            metadata: {
                priority, description,
                provenance: `RuleManager (${shortId(trigger.id)})`,
                created: new Date().toISOString()
            }
        };
        context.rules.add(newRule);
        log('warn', `Rule ${shortId(newRule.id)} added programmatically by ${shortId(trigger.id)}`);
        return this.createResultAtom(true, "rule_added", shortId(newRule.id));
    }

    private deleteRule(action: Struct, context: ToolContext): Term {
        const ruleIdTerm = action.args[1];
        if (ruleIdTerm?.kind !== 'Atom') return this.createResultAtom(false, "invalid_delete_rule_params");
        const rule = context.rules.findRule(ruleIdTerm.name); // Use prefix finder
        if (!rule) return this.createResultAtom(false, "rule_not_found", ruleIdTerm.name);

        context.rules.delete(rule.id);
        log('warn', `Rule ${shortId(rule.id)} deleted programmatically.`);
        return this.createResultAtom(true, "rule_deleted", shortId(rule.id));
    }

    private modifyPriority(action: Struct, context: ToolContext): Term {
        const ruleIdTerm = action.args[1];
        const deltaTerm = action.args[2];
        if (ruleIdTerm?.kind !== 'Atom' || deltaTerm?.kind !== 'Atom' || !/^-?\d+$/.test(deltaTerm.name)) {
            return this.createResultAtom(false, "invalid_modify_priority_params");
        }
        const rule = context.rules.findRule(ruleIdTerm.name); // Use prefix finder
        if (!rule) return this.createResultAtom(false, "rule_not_found", ruleIdTerm.name);

        const delta = parseInt(deltaTerm.name, 10);
        const oldPriority = rule.metadata.priority ?? 0;
        rule.metadata.priority = oldPriority + delta;
        context.rules.update(rule);
        log('warn', `Rule ${shortId(rule.id)} priority changed from ${oldPriority} to ${rule.metadata.priority}`);
        return this.createResultAtom(true, "priority_modified", `${shortId(rule.id)} to ${rule.metadata.priority}`);
    }
}

class Tools {
    private tools = new Map<string, Tool>();
    register(tool: Tool): void {
        if (this.tools.has(tool.name)) log('warn', `Tool "${tool.name}" redefined.`);
        this.tools.set(tool.name, tool);
    }
    get(name: string): Tool | undefined { return this.tools.get(name); }
    list(): Tool[] { return Array.from(this.tools.values()); }
}

// --- Engine Components ---
const RuleMatcher = {
    match(thought: Thought, rules: Rule[]): MatchResult | null {
        const applicableMatches: MatchResult[] = [];
        for (const rule of rules) {
            try {
                const bindings = Terms.unify(rule.pattern, thought.content);
                if (bindings !== null) {
                    applicableMatches.push({ rule, bindings });
                }
            } catch (e: any) {
                // Log unification errors but don't halt matching
                log('warn', `Unification error matching rule ${shortId(rule.id)} against thought ${shortId(thought.id)}: ${e.message}`);
            }
        }

        const count = applicableMatches.length;
        if (count === 0) return null;
        if (count === 1) return applicableMatches[0];

        // Sort by priority (desc), then belief score (desc), then creation time (desc - newer first)
        applicableMatches.sort((a, b) =>
            (b.rule.metadata.priority ?? 0) - (a.rule.metadata.priority ?? 0) ||
            b.rule.belief.score() - a.rule.belief.score() ||
            (b.rule.metadata.created ?? '').localeCompare(a.rule.metadata.created ?? '')
        );
        log('debug', `Multiple rule matches for ${shortId(thought.id)} (${thought.type}). Best: ${shortId(applicableMatches[0].rule.id)} (Priority: ${applicableMatches[0].rule.metadata.priority ?? 0}, Score: ${applicableMatches[0].rule.belief.score().toFixed(2)})`);
        return applicableMatches[0];
    }
};

const ActionExecutor = {
    async execute(actionTerm: Term, context: ToolContext, trigger: Thought, rule?: Rule): Promise<ActionResult> {
        if (actionTerm.kind !== 'Struct') {
            return { success: false, error: `Invalid action term kind: ${actionTerm.kind}`, finalStatus: Status.FAILED };
        }
        const tool = context.tools.get(actionTerm.name);
        if (!tool) {
            return { success: false, error: `Tool not found: ${actionTerm.name}`, finalStatus: Status.FAILED };
        }

        log('debug', `Executing action ${Terms.format(actionTerm, false)} via tool ${tool.name} for thought ${shortId(trigger.id)} (Rule: ${shortId(rule?.id)})`);
        try {
            // Execute the tool action
            const resultTerm = await tool.execute(actionTerm, context, trigger);

            // IMPORTANT: Re-fetch the thought state as the tool might have modified it (e.g., set status to WAITING)
            const currentThoughtState = context.thoughts.get(trigger.id);
            if (!currentThoughtState) {
                log('warn', `Thought ${shortId(trigger.id)} was deleted during action execution by ${tool.name}. Assuming success.`);
                return { success: true }; // If deleted, action implicitly succeeded in its goal?
            }

            // Determine outcome based on the *current* state and tool result
            const isWaiting = currentThoughtState.status === Status.WAITING;
            const isFailedByTool = currentThoughtState.status === Status.FAILED; // Tool explicitly set to FAILED
            const returnedError = resultTerm?.kind === 'Atom' && resultTerm.name.startsWith('error:');

            if (returnedError) {
                // Tool returned an error atom
                return { success: false, error: `Tool failed: ${resultTerm.name}`, finalStatus: Status.FAILED };
            } else if (isWaiting) {
                // Tool put the thought into WAITING state (e.g., UserInteractionTool)
                return { success: true, finalStatus: Status.WAITING };
            } else if (isFailedByTool) {
                // Tool explicitly marked the thought as FAILED
                return { success: false, error: `Tool set status to FAILED`, finalStatus: Status.FAILED };
            } else {
                // Default success case: Tool completed without error and didn't set WAITING/FAILED
                return { success: true, finalStatus: Status.DONE };
            }
        } catch (error: any) {
            // Catch exceptions thrown *by* the tool's execute method
            log('error', `Tool exception in ${tool.name} processing ${shortId(trigger.id)}: ${error.message}`, error.stack);
            return { success: false, error: `Tool exception: ${error.message}`, finalStatus: Status.FAILED };
        }
    }
};

const FallbackHandler = {
    async handle(thought: Thought, context: ToolContext): Promise<FallbackResult> {
        log('debug', `Applying fallback handler for thought ${shortId(thought.id)} (${thought.type})`);
        let action: Term | null = null;
        let llmPromptKey: PromptKey | null = null;
        let llmContext: Record<string, string> = {};
        let targetType: Type | null = null;
        const contentStr = Terms.toString(thought.content).substring(0, 100); // Limit content for prompts

        // Define fallback behaviors based on thought type
        switch (thought.type) {
            case Type.INPUT:
                llmPromptKey = 'GENERATE_GOAL'; llmContext = { input: contentStr }; targetType = Type.GOAL; break;
            case Type.GOAL:
                llmPromptKey = 'GENERATE_STRATEGY'; llmContext = { goal: contentStr }; targetType = Type.STRATEGY; break;
            case Type.STRATEGY:
                llmPromptKey = 'GENERATE_OUTCOME'; llmContext = { strategy: contentStr }; targetType = Type.OUTCOME; break;
            case Type.OUTCOME: case Type.FACT:
                // Default for facts/outcomes: try to add to memory
                action = Struct("MemoryTool", [Atom("add"), thought.content]); break;
            case Type.LOG: case Type.SYSTEM:
                // Logs and System ticks usually don't need fallbacks, just mark done
                return { success: true, finalStatus: Status.DONE };
            default:
                // For unknown or unhandled types, ask the user
                const askUserContext = { thoughtId: shortId(thought.id), thoughtType: thought.type, thoughtContent: contentStr };
                action = Struct("UserInteractionTool", [
                    Atom("prompt"),
                    Atom("FALLBACK_ASK_USER"),
                    Struct("context", Object.entries(askUserContext).map(([k, v]) => Atom(`${k}:${v}`))) // Create context struct
                ]);
                break;
        }

        // Execute LLM generation if defined
        if (llmPromptKey && targetType) {
            try {
                const prompt = context.prompts.format(llmPromptKey, llmContext);
                const resultText = await context.llm.generate(prompt);
                // Process multi-line results (e.g., strategies)
                resultText.split('\n')
                    .map(s => s.trim().replace(/^- /, '')) // Clean up lines
                    .filter(s => s.length > 0)
                    .forEach(resText => {
                        context.engine.addThought({
                            id: generateId(), type: targetType!, content: Atom(resText), belief: Belief.DEFAULT, status: Status.PENDING,
                            metadata: {
                                rootId: thought.metadata.rootId ?? thought.id, parentId: thought.id,
                                created: new Date().toISOString(), provenance: 'llm_fallback'
                            }
                        });
                    });
                return { success: true, finalStatus: Status.DONE };
            } catch (error: any) {
                return { success: false, error: `LLM fallback failed: ${error.message}`, finalStatus: Status.FAILED };
            }
        }
        // Execute action if defined
        else if (action) {
            return ActionExecutor.execute(action, context, thought); // Execute the fallback action
        }
        // If no LLM or action defined
        else {
            return { success: false, error: "No rule match and no fallback action defined.", finalStatus: Status.FAILED };
        }
    }
};

// --- Engine ---
class Engine {
    private activeIds = new Set<string>(); // Tracks thoughts currently being processed
    private context: ToolContext;

    constructor(
        private thoughts: Thoughts, private rules: Rules, private memory: Memory,
        private llm: LLM, private tools: Tools, private prompts: Prompts,
        private batchSize: number = 5, // Max thoughts to *start* processing per cycle
        private maxConcurrent: number = 3 // Max thoughts actively processing at once
    ) {
        this.context = { thoughts: this.thoughts, rules: this.rules, memory: this.memory, llm: this.llm, tools: this.tools, prompts: this.prompts, engine: this };
    }

    addThought(thought: Thought): void { this.thoughts.add(thought); }

    // Selects the next thought to process based on status, task status, and priority/belief
    private sampleThought(): Thought | null {
        const candidates = this.thoughts.getPending().filter(t => {
            // Don't process if already active
            if (this.activeIds.has(t.id)) return false;
            // Check task status (only process if root task is RUNNING)
            const rootId = t.metadata.rootId ?? t.id;
            const rootThought = this.thoughts.get(rootId);
            // If root doesn't exist or is PAUSED, skip this thought
            return !rootThought || rootThought.metadata.taskStatus !== 'PAUSED';
        });

        if (candidates.length === 0) return null;

        // Weighted random sampling based on priority (higher is better) and belief score
        // Give a small base weight to ensure even low-priority items can eventually run
        const weights = candidates.map(t => Math.max(0.01, (t.metadata.priority ?? 0) + t.belief.score()));
        const totalWeight = weights.reduce((sum, w) => sum + w, 0);

        // Handle edge case where all weights are zero or negative
        if (totalWeight <= 0) return candidates[Math.floor(Math.random() * candidates.length)];

        let random = Math.random() * totalWeight;
        for (let i = 0; i < candidates.length; i++) {
            random -= weights[i];
            if (random <= 0) return candidates[i];
        }
        // Fallback in case of floating point issues
        return candidates[candidates.length - 1];
    }

    // Updates thought status, belief, metadata based on processing result
    private updateThoughtStatus(thought: Thought, result: ActionResult | FallbackResult, rule?: Rule): void {
        // Re-fetch thought in case it was modified during processing (e.g., by a tool)
        const currentThought = this.thoughts.get(thought.id);
        if (!currentThought) {
            log('warn', `Thought ${shortId(thought.id)} no longer exists after processing.`);
            return; // Nothing to update
        }

        // Update belief based on success/failure
        currentThought.belief.update(result.success);
        if (rule) {
            currentThought.metadata.ruleId = rule.id; // Record which rule was applied
        }

        if (result.success) {
            currentThought.status = result.finalStatus ?? Status.DONE; // Default to DONE on success
            delete currentThought.metadata.error; // Clear previous errors
            delete currentThought.metadata.retries; // Clear retries
            // Keep waitingFor if status is WAITING, otherwise clear it
            if (currentThought.status !== Status.WAITING) {
                delete currentThought.metadata.waitingFor;
            }
        } else {
            // Handle failure
            const retries = (currentThought.metadata.retries ?? 0) + 1;
            currentThought.status = retries >= MAX_RETRIES ? Status.FAILED : Status.PENDING; // Retry or mark failed
            currentThought.metadata.error = (result.error ?? "Unknown processing error").substring(0, 250);
            currentThought.metadata.retries = retries;
            log('warn', `Thought ${shortId(currentThought.id)} failed (Attempt ${retries}/${MAX_RETRIES}): ${currentThought.metadata.error}`);

// Log the failure event itself if it finally failed (and isn't already a log)
if (currentThought.status === Status.FAILED && currentThought.type !== Type.LOG) {
    this.addThought({
        id: generateId(), type: Type.LOG,
        content: Struct("failure", [
            Struct("thoughtId", [Atom(currentThought.id)]),
            Struct("content", [currentThought.content]), // Include original content for context
            Struct("error", [Atom(currentThought.metadata.error ?? "Unknown")])
        ]),
        belief: Belief.DEFAULT, status: Status.PENDING,
        metadata: {
            rootId: currentThought.metadata.rootId ?? currentThought.id,
            parentId: currentThought.id,
            created: new Date().toISOString(),
            provenance: 'engine_failure_log'
        }
    });
}
}
this.thoughts.update(currentThought); // Persist changes
}

// Apply a matched rule
private async _applyRule(match: MatchResult, thought: Thought): Promise<ActionResult> {
    const boundAction = Terms.substitute(match.rule.action, match.bindings);
    const result = await ActionExecutor.execute(boundAction, this.context, thought, match.rule);
    // Update rule belief based on action success/failure
    match.rule.belief.update(result.success);
    this.rules.update(match.rule);
    return result;
}

// Apply fallback logic when no rule matches
private async _applyFallback(thought: Thought): Promise<FallbackResult> {
    return await FallbackHandler.handle(thought, this.context);
}

// Core processing logic for a single thought
private async _processThought(thought: Thought): Promise<boolean> {
    // Mark as active
    this.activeIds.add(thought.id);
    try {
        log('info', `Processing ${thought.type} ${shortId(thought.id)}: ${Terms.format(thought.content, false).substring(0, 80)}`);
thought.status = Status.ACTIVE;
thought.metadata.agentId = 'worker'; // Mark who processed it
this.thoughts.update(thought); // Update status immediately

// Find matching rule
const match = RuleMatcher.match(thought, this.rules.getAll());
let result: ActionResult | FallbackResult;
let appliedRule: Rule | undefined = undefined;

// Execute rule or fallback
if (match) {
    appliedRule = match.rule;
    result = await this._applyRule(match, thought);
} else {
    result = await this._applyFallback(thought);
}

// Update thought status based on result
this.updateThoughtStatus(thought, result, appliedRule);

// Return true if processing is complete (not PENDING or ACTIVE)
const finalStatus = this.thoughts.get(thought.id)?.status;
return finalStatus !== Status.PENDING && finalStatus !== Status.ACTIVE;

} catch (error: any) {
    log('error', `Critical error processing ${shortId(thought.id)}: ${error.message}`, error.stack);
    // Ensure status is updated even on unexpected errors
    this.updateThoughtStatus(thought, { success: false, error: `Unhandled exception: ${error.message}`, finalStatus: Status.FAILED });
    return true; // Completed (with critical error)
} finally {
    // Always remove from active set when processing finishes (or errors out)
    this.activeIds.delete(thought.id);
    // Sanity check: Ensure thought didn't somehow end up ACTIVE
    const finalThoughtState = this.thoughts.get(thought.id);
    if (finalThoughtState?.status === Status.ACTIVE) {
        log('warn', `Thought ${shortId(thought.id)} processing finished but status remained ACTIVE. Setting FAILED.`);
        this.updateThoughtStatus(finalThoughtState, { success: false, error: "Processing ended while ACTIVE.", finalStatus: Status.FAILED });
    }
}
}

// Process a batch of thoughts respecting concurrency limits
async processBatch(): Promise<number> {
    const promises: Promise<boolean>[] = [];
let acquiredCount = 0;
// Acquire thoughts up to batchSize, respecting maxConcurrent
while (this.activeIds.size < this.maxConcurrent && acquiredCount < this.batchSize) {
    const thought = this.sampleThought();
    if (!thought) break; // No more eligible thoughts
    // Start processing, don't await here
    promises.push(this._processThought(thought));
    acquiredCount++;
}

if (promises.length === 0) return 0; // Nothing was processed

// Wait for all started processes in this batch to complete
await Promise.all(promises);
log('debug', `Processed batch of ${promises.length} thoughts.`);
return promises.length; // Return how many were processed
}

// Handle response submitted by user for a specific prompt
async handlePromptResponse(promptId: string, responseText: string): Promise<boolean> {
    // Find the original USER_PROMPT thought that is PENDING
    const promptThought = this.thoughts.findPendingPrompt(promptId);
    if (!promptThought) {
    log('error', `Pending prompt thought for ID ${shortId(promptId)} not found or already processed.`);
    return false;
}

// Find the thought that was WAITING for this promptId
const waitingThought = this.thoughts.findWaitingThought(promptId);
if (!waitingThought) {
    log('warn', `No thought found waiting for prompt ${shortId(promptId)}. Marking prompt as done but taking no further action.`);
    promptThought.status = Status.DONE;
    promptThought.metadata.error = "No waiting thought found upon response.";
    this.thoughts.update(promptThought);
    return false; // Indicate response wasn't linked to a waiting thought
}

// 1. Create a new INPUT thought representing the user's response
this.addThought({
    id: generateId(),
    type: Type.INPUT,
    content: Terms.fromString(responseText), // Parse response as Term
    belief: new Belief(1, 0), // User input is generally trusted initially
    status: Status.PENDING,
    metadata: {
        rootId: waitingThought.metadata.rootId ?? waitingThought.id, // Link to same task
        parentId: waitingThought.id, // The waiting thought is the parent
        created: new Date().toISOString(),
        responseTo: promptId, // Link back to the prompt ID
        tags: ['user_response'],
        provenance: 'user_input'
    }
});

// 2. Mark the original USER_PROMPT thought as DONE
promptThought.status = Status.DONE;
this.thoughts.update(promptThought);

// 3. Reactivate the waiting thought
waitingThought.status = Status.PENDING;
delete waitingThought.metadata.waitingFor; // Clear the waiting condition
waitingThought.belief.update(true); // User responding implies the wait was 'successful'
this.thoughts.update(waitingThought);

log('info', `Response for ${shortId(promptId)} received. Reactivated thought ${shortId(waitingThought.id)}.`);
return true;
}
}

// --- Persistence ---
const saveState = async (thoughts: Thoughts, rules: Rules, memory: Memory): Promise<void> => {
    try {
        await fs.mkdir(DATA_DIR, { recursive: true });
        const state: AppState = { thoughts: thoughts.toJSON(), rules: rules.toJSON() };
        await fs.writeFile(STATE_FILE, JSON.stringify(state, null, 2));
        // Also save memory store if it's ready
        if (memory.isReady) await memory.save();
        log('debug', 'State saved successfully.');
    } catch (error: any) { log('error', 'Error saving state:', error.message); }
};
// Debounced version for frequent updates
const debouncedSaveState = debounce(saveState, SAVE_DEBOUNCE);

const loadState = async (thoughts: Thoughts, rules: Rules, memory: Memory): Promise<void> => {
    try {
        await fs.mkdir(DATA_DIR, { recursive: true });
        // Initialize memory first (creates dir if needed)
        await memory.initialize();
        if (!memory.isReady) log('error', "Memory store failed to initialize. State loading might be incomplete.");

        // Load thoughts and rules from JSON file if it exists
        if (fsSync.existsSync(STATE_FILE)) {
            const data = await fs.readFile(STATE_FILE, 'utf-8');
            const state: AppState = safeJsonParse(data, { thoughts: {}, rules: {} });
            thoughts.loadJSON(state.thoughts); // Handles parsing and validation
            rules.loadJSON(state.rules);     // Handles parsing and validation
        } else {
            log('warn', `State file ${STATE_FILE} not found. Starting with empty thoughts/rules state.`);
        }
    } catch (error: any) {
        log('error', 'Error loading state:', error.message);
        // Attempt memory init again if file loading failed, just in case
        if (!memory.isReady) await memory.initialize();
    }
};

// --- WebSocket API Server ---
class ApiServer {
    private wss: WebSocketServer | null = null;
    private clients = new Map<WebSocket, { ip: string }>();
    private commandHandlers = new Map<string, (payload: any, ws: WebSocket) => Promise<any | void>>();

    constructor(
        private port: number, private thoughts: Thoughts, private rules: Rules,
        private memory: Memory, private engine: Engine, private tools: Tools,
        private systemControl: SystemControl
    ) {
        this.registerCommandHandlers();
        // Listen for changes in stores to broadcast deltas
        this.thoughts.addChangeListener(() => this.broadcastChanges(this.thoughts, 'thoughts'));
        this.rules.addChangeListener(() => this.broadcastChanges(this.rules, 'rules'));
    }

    start(): void {
        if (this.wss) { log('warn', 'API Server already started.'); return; }
        this.wss = new WebSocketServer({ port: this.port });
        this.wss.on('listening', () => log('info', `🚀 FlowMind API listening on ws://localhost:${this.port}`));
        this.wss.on('connection', (ws, req) => this.handleConnection(ws, req as IncomingMessage));
        this.wss.on('error', (error) => log('error', 'WebSocket Server Error:', error.message));
    }

    stop(): void {
        if (!this.wss) return;
        log('warn', 'Stopping API server...');
        this.wss.clients.forEach(ws => ws.close(1000, 'Server shutting down')); // Close connections gracefully
        this.wss.close((err) => {
            if (err) log('error', 'Error closing WebSocket server:', err.message);
            else log('info', 'API Server stopped.');
        });
        this.clients.clear();
        this.wss = null;
    }

    private handleConnection(ws: WebSocket, req: IncomingMessage): void {
        const ip = req.socket.remoteAddress || req.headers['x-forwarded-for']?.toString() || 'unknown';
        log('info', `Client connected: ${ip}`);
        this.clients.set(ws, { ip });

        ws.on('message', (message) => this.handleMessage(ws, message as Buffer, ip));
        ws.on('close', (code, reason) => this.handleDisconnection(ws, ip, code, reason.toString()));
        ws.on('error', (error) => { log('error', `WebSocket error from ${ip}: ${error.message}`); this.handleDisconnection(ws, ip, 1011, 'WebSocket error'); }); // 1011: Internal Error

        // Send initial state snapshot to the new client
        this.sendSnapshot(ws);
    }

    private handleDisconnection(ws: WebSocket, ip: string, code: number, reason: string): void {
        log('info', `Client disconnected: ${ip} (Code: ${code}, Reason: ${reason || 'N/A'})`);
        this.clients.delete(ws);
    }

    private async handleMessage(ws: WebSocket, message: Buffer, ip: string): Promise<void> {
        let request: { command: string; payload?: any; requestId?: string } | null = null;
        try {
            request = JSON.parse(message.toString());
            if (typeof request?.command !== 'string') throw new Error("Invalid command format: 'command' field missing or not a string.");
        } catch (e: any) {
            this.send(ws, { type: 'error', payload: `Invalid JSON request: ${e.message}`, requestId: request?.requestId });
            return;
        }

        const handler = this.commandHandlers.get(request.command);
        if (!handler) {
            this.send(ws, { type: 'error', payload: `Unknown command: ${request.command}`, requestId: request.requestId });
            return;
        }

        try {
            log('debug', `Executing command "${request.command}" from client ${ip}`);
            const result = await handler(request.payload ?? {}, ws); // Pass payload, default to empty object
            // Send success response (payload can be null/undefined)
            this.send(ws, { type: 'response', payload: result, success: true, requestId: request.requestId });

            // Trigger save state for commands that modify state (exclude read-only commands)
            const readOnlyCommands = ['thoughts', 'rules', 'tasks', 'task', 'status', 'info', 'search', 'help', 'tools', 'ping'];
            if (!readOnlyCommands.includes(request.command)) {
                debouncedSaveState(this.thoughts, this.rules, this.memory);
            }
        } catch (error: any) {
            log('error', `Error executing command "${request.command}" from ${ip}: ${error.message}`);
            // Send failure response
            this.send(ws, { type: 'response', payload: error.message || 'Command failed.', success: false, requestId: request.requestId });
        }
    }

    // Send data to a specific client
    private send(ws: WebSocket, data: WsMessage): void {
        if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify(data));
        } else {
            log('warn', `Attempted to send to client in state ${ws.readyState}`);
        }
    }

    // Send data to all connected clients
    broadcast(data: WsMessage): void {
        if (this.clients.size === 0) return;
        log('debug', `Broadcasting message type ${data.type} to ${this.clients.size} client(s)`);
        const message = JSON.stringify(data);
        this.clients.forEach((_, ws) => {
            if (ws.readyState === WebSocket.OPEN) ws.send(message);
        });
    }

    // Broadcasts only the changed/deleted items since the last broadcast
    private broadcastChanges(store: BaseStore<any>, type: 'thoughts' | 'rules'): void {
        const delta = store.getDelta();
        if (delta.changed.length > 0 || delta.deleted.length > 0) {
            // Serialize belief for thoughts/rules before broadcasting
            const serializeItem = (item: any) => ({ ...item, belief: item.belief?.toJSON() });
            const payload = {
                changed: delta.changed.map(serializeItem),
                deleted: delta.deleted
            };
            this.broadcast({ type: `${type}_delta`, payload });
            log('debug', `Broadcasting ${type} delta: ${delta.changed.length} changed, ${delta.deleted.length} deleted`);
        }
    }

    // Sends the complete current state to a newly connected client
    private sendSnapshot(ws: WebSocket): void {
        const serializeItem = (item: any) => ({ ...item, belief: item.belief?.toJSON() });
        this.send(ws, {
            type: 'snapshot', payload: {
                thoughts: this.thoughts.getAll().map(serializeItem),
                rules: this.rules.getAll().map(serializeItem),
                status: this.systemControl.getStatus()
            }
        });
        log('debug', `Sent snapshot to new client`);
    }

    // --- Command Handler Helpers ---
    private _findThoughtOrThrow(idPrefix: string, requireRoot: boolean = false): Thought {
        const thought = this.thoughts.findThought(idPrefix);
        if (!thought) throw new Error(`Thought prefix "${idPrefix}" not found or ambiguous.`);
        if (requireRoot && (thought.metadata.rootId && thought.metadata.rootId !== thought.id)) {
            throw new Error(`Thought ${shortId(thought.id)} (${thought.type}) is not a root task.`);
        }
        return thought;
    }

    private _findRuleOrThrow(idPrefix: string): Rule {
        const rule = this.rules.findRule(idPrefix);
        if (!rule) throw new Error(`Rule prefix "${idPrefix}" not found or ambiguous.`);
        return rule;
    }

    // --- Command Handlers ---
    private registerCommandHandlers(): void {
        const handlers: { [key: string]: (payload: any, ws: WebSocket) => Promise<any | void> } = {
            add: async ({ text }) => {
                if (typeof text !== 'string' || !text.trim()) throw new Error("Payload requires non-empty 'text' field.");
                const newThought: Thought = {
                    id: generateId(), type: Type.INPUT, content: Terms.fromString(text), belief: new Belief(1, 0), status: Status.PENDING,
                    metadata: { created: new Date().toISOString(), tags: ['user_added'], provenance: 'api', taskStatus: 'RUNNING' }
                };
                newThought.metadata.rootId = newThought.id; // It's its own root
                this.engine.addThought(newThought);
                return { id: newThought.id, message: `Added Task ${newThought.type}: ${shortId(newThought.id)}` };
            },
            respond: async ({ promptId, text }) => {
                if (typeof promptId !== 'string' || !promptId.trim()) throw new Error("Payload requires 'promptId' field.");
                if (typeof text !== 'string' || !text.trim()) throw new Error("Payload requires non-empty 'text' field.");
                const success = await this.engine.handlePromptResponse(promptId, text);
                if (!success) throw new Error(`Failed to process response for prompt ${shortId(promptId)}. Check logs for details.`);
                return { message: `Response for ${shortId(promptId)} processed.` };
            },
            run: async () => { this.systemControl.startProcessing(); return { message: "Processing started." }; },
            pause: async () => { this.systemControl.pauseProcessing(); return { message: "Processing paused." }; },
            step: async () => {
                const count = await this.systemControl.stepProcessing();
                return { processed: count > 0, count, message: count > 0 ? `Step processed ${count} thought(s).` : "Nothing to step." };
            },
            save: async () => {
                await debouncedSaveState.flush(); // Ensure debounced save finishes
                await saveState(this.thoughts, this.rules, this.memory); // Force immediate save
                return { message: "State saved." };
            },
            quit: async () => { await this.systemControl.shutdown(); }, // Handled by System.shutdown
            status: async () => this.systemControl.getStatus(),
            info: async ({ idPrefix }) => {
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                const thought = this.thoughts.findThought(idPrefix);
                if (thought) return { type: 'thought', data: { ...thought, belief: thought.belief.toJSON() } };
                const rule = this.rules.findRule(idPrefix);
                if (rule) return { type: 'rule', data: { ...rule, belief: rule.belief.toJSON() } };
                throw new Error(`Item prefix "${idPrefix}" not found or ambiguous.`);
            },
            thoughts: async () => this.thoughts.getAll().map(t => ({ ...t, belief: t.belief.toJSON() })),
            rules: async () => this.rules.getAll().map(r => ({ ...r, belief: r.belief.toJSON() })),
            tools: async () => this.tools.list().map(t => ({ name: t.name, description: t.description })),
            tag: async ({ idPrefix, tag }) => {
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                if (typeof tag !== 'string' || !tag.trim()) throw new Error("Payload requires non-empty 'tag' field.");
                const thought = this._findThoughtOrThrow(idPrefix); // Find any thought
                const cleanTag = tag.trim();
                thought.metadata.tags = [...new Set([...(thought.metadata.tags || []), cleanTag])];
                this.thoughts.update(thought);
                return { id: thought.id, tags: thought.metadata.tags };
            },
            untag: async ({ idPrefix, tag }) => {
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                if (typeof tag !== 'string' || !tag.trim()) throw new Error("Payload requires non-empty 'tag' field.");
                const thought = this._findThoughtOrThrow(idPrefix);
                if (!thought.metadata.tags) return { id: thought.id, tags: [] }; // No tags to remove from
                const cleanTag = tag.trim();
                thought.metadata.tags = thought.metadata.tags.filter(t => t !== cleanTag);
                this.thoughts.update(thought);
                return { id: thought.id, tags: thought.metadata.tags };
            },
            search: async ({ query, k }) => {
                if (typeof query !== 'string' || !query.trim()) throw new Error("Payload requires non-empty 'query' field.");
                const numResults = (typeof k === 'number' && k > 0) ? Math.floor(k) : 5; // Default k=5
                return this.memory.search(query, numResults);
            },
            delete: async ({ idPrefix }) => {
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                // Try deleting thought first
                const thought = this.thoughts.findThought(idPrefix);
                if (thought) {
                    const deleted = this.thoughts.delete(thought.id);
                    if (deleted) return { id: thought.id, type: 'thought', message: `Thought ${shortId(thought.id)} deleted.` };
                    else throw new Error(`Failed to delete thought ${shortId(thought.id)}.`);
                }
                // Try deleting rule if thought not found
                const rule = this.rules.findRule(idPrefix);
                if (rule) {
                    const deleted = this.rules.delete(rule.id);
                    if (deleted) return { id: rule.id, type: 'rule', message: `Rule ${shortId(rule.id)} deleted.` };
                    else throw new Error(`Failed to delete rule ${shortId(rule.id)}.`);
                }
                // If neither found
                throw new Error(`Item with prefix "${idPrefix}" not found or ambiguous.`);
            },
            tasks: async () => { // List root thoughts (tasks)
                return this.thoughts.getRootThoughts().map(task => {
                    const descendants = this.thoughts.getDescendants(task.id);
                    const pendingPrompts = descendants.filter(t => t.type === Type.USER_PROMPT && t.status === Status.PENDING);
                    return {
                        id: task.id, type: task.type, content: Terms.toString(task.content),
                        status: task.status, taskStatus: task.metadata.taskStatus ?? 'RUNNING', // Default to RUNNING if missing
                        pendingPrompts: pendingPrompts.map(p => ({ id: p.id, content: Terms.toString(p.content) }))
                    };
                });
            },
            task: async ({ idPrefix }) => { // Get details for a specific task (root thought)
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                const task = this._findThoughtOrThrow(idPrefix, true); // Ensure it's a root task
                const descendants = this.thoughts.getDescendants(task.id);
                // Helper to format thought details concisely
                const format = (thoughts: Thought[]) => thoughts
                    .sort((a, b) => (a.metadata.created ?? '').localeCompare(b.metadata.created ?? '')) // Sort chronologically
                    .map(t => ({
                        id: t.id, type: t.type, content: Terms.toString(t.content),
                        status: t.status, created: t.metadata.created
                    }));
                return {
                    task: { ...task, belief: task.belief.toJSON() }, // Include main task details
                    plan: format(descendants.filter(t => t.type === Type.STRATEGY)),
                    outcomes: format(descendants.filter(t => t.type === Type.OUTCOME)),
                    prompts: format(descendants.filter(t => t.type === Type.USER_PROMPT)),
                    facts: format(descendants.filter(t => t.type === Type.FACT)),
                    logs: format(descendants.filter(t => t.type === Type.LOG)),
                    // Include other types just in case
                    other: format(descendants.filter(t => ![Type.STRATEGY, Type.OUTCOME, Type.USER_PROMPT, Type.FACT, Type.LOG].includes(t.type))),
                };
            },
            pause_task: async ({ idPrefix }) => {
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                const task = this._findThoughtOrThrow(idPrefix, true); // Ensure it's a root task
                task.metadata.taskStatus = 'PAUSED';
                this.thoughts.update(task);
                return { id: task.id, taskStatus: task.metadata.taskStatus, message: `Task ${shortId(task.id)} paused.` };
            },
            run_task: async ({ idPrefix }) => {
                if (typeof idPrefix !== 'string' || !idPrefix.trim()) throw new Error("Payload requires 'idPrefix' field.");
                const task = this._findThoughtOrThrow(idPrefix, true); // Ensure it's a root task
                task.metadata.taskStatus = 'RUNNING';
                this.thoughts.update(task);
                return { id: task.id, taskStatus: task.metadata.taskStatus, message: `Task ${shortId(task.id)} set to running.` };
            },
            help: async () => ({
                description: "Available commands. Use 'info <idPrefix>' for details on specific thoughts/rules.",
                commands: [
                    { cmd: "add <text>", desc: "Add new root INPUT thought (creates a new Task)." },
                    { cmd: "respond <promptId> <text>", desc: "Respond to a pending user prompt." },
                    { cmd: "run", desc: "Start global processing loop." },
                    { cmd: "pause", desc: "Pause global processing loop." },
                    { cmd: "step", desc: "Process one batch of thoughts (respects task status)." },
                    { cmd: "save", desc: "Force immediate state save to disk." },
                    { cmd: "quit / exit", desc: "Shutdown the server (from REPL only)." },
                    { cmd: "status", desc: "Get global processing loop status (RUNNING/PAUSED)." },
                    { cmd: "info <idPrefix>", desc: "Get full details of a thought or rule by ID prefix." },
                    { cmd: "thoughts", desc: "Get all thoughts (can be large)." },
                    { cmd: "rules", desc: "Get all rules." },
                    { cmd: "tasks", desc: "List all root tasks and their status/pending prompts." },
                    { cmd: "task <taskIdPrefix>", desc: "Get detailed view of a specific task (plan, facts, logs, etc.)." },
                    { cmd: "run_task <taskIdPrefix>", desc: "Set a specific task to RUNNING state." },
                    { cmd: "pause_task <taskIdPrefix>", desc: "Set a specific task to PAUSED state." },
                    { cmd: "tools", desc: "List available tools and their descriptions." },
                    { cmd: "tag <idPrefix> <tag>", desc: "Add a tag to a thought." },
                    { cmd: "untag <idPrefix> <tag>", desc: "Remove a tag from a thought." },
                    { cmd: "search <query> [k=5]", desc: "Search vector memory for relevant entries." },
                    { cmd: "delete <idPrefix>", desc: "Delete a thought or rule by ID prefix." },
                    { cmd: "ping", desc: "Check server connection." },
                    { cmd: "help", desc: "Show this help message." },
                ]
            }),
            ping: async () => "pong",
        };
        // Register handlers, binding 'this' context
        for (const [command, handler] of Object.entries(handlers)) {
            this.commandHandlers.set(command, handler.bind(this));
        }
    }
}

// --- Main System Orchestrator ---
class System {
    private thoughts = new Thoughts();
    private rules = new Rules();
    private llm = new LLM();
    private memory = new Memory(this.llm.embeddings, VECTOR_STORE_DIR);
    private tools = new Tools();
    private prompts = new Prompts();
    private engine: Engine;
    private apiServer: ApiServer;
    private isRunning = false;
    private workerIntervalId: NodeJS.Timeout | null = null;
    private systemControl: SystemControl;

    constructor() {
        this.engine = new Engine(this.thoughts, this.rules, this.memory, this.llm, this.tools, this.prompts);
        this.addTools(); // Register tools first
        this.systemControl = this.createSystemControl();
        this.apiServer = new ApiServer(WS_PORT, this.thoughts, this.rules, this.memory, this.engine, this.tools, this.systemControl);
    }

    private addTools(): void {
        [
            new LLMTool(), new MemoryTool(), new UserInteractionTool(), new GoalProposalTool(), new CoreTool(),
            new TimeTool(), new FileSystemTool(), new WebSearchTool(), new ContextSummarizer(), new RuleManager()
        ].forEach(t => this.tools.register(t));
    }

    private createSystemControl(): SystemControl {
        return {
            startProcessing: this.startProcessingLoop.bind(this),
            pauseProcessing: this.pauseProcessingLoop.bind(this),
            stepProcessing: this.engine.processBatch.bind(this.engine), // Delegate step to engine
            shutdown: this.shutdown.bind(this),
            getStatus: () => ({ isRunning: this.isRunning })
        };
    }

    // Adds default rules if none exist
    private addBootstrapRules(): void {
        if (this.rules.count() > 0) {
            log('debug', "Skipping bootstrap rules: Rules already loaded.");
            return;
        }
        log('info', "Bootstrapping default rules...");
        const rule = (desc: string, pattern: Term, action: Term, priority: number = 0): Rule => ({
            id: generateId(), pattern, action, belief: Belief.DEFAULT,
            metadata: { description: desc, provenance: 'bootstrap', created: new Date().toISOString(), priority }
        });

        const rulesToAdd = [
            // --- Core Flow & Memory ---
            rule("Outcome -> Add to Memory", Struct(Type.OUTCOME, [Variable("C")]), Struct("MemoryTool", [Atom("add"), Variable("C")]), 10),
            rule("Fact -> Add to Memory", Struct(Type.FACT, [Variable("C")]), Struct("MemoryTool", [Atom("add"), Variable("C")]), 10),
            rule("Outcome -> Log it", Struct(Type.OUTCOME, [Variable("OutcomeContent")]), Struct("CoreTool", [Atom("add_thought"), Atom(Type.LOG), Struct("logged_outcome", [Variable("OutcomeContent")]), Atom("self"), Atom("self")]), -5),
            // rule("Outcome -> Suggest Next Goal", Struct(Type.OUTCOME, [Variable("C")]), Struct("GoalProposalTool", [Atom("suggest"), Variable("C")]), -10), // Maybe too noisy?

            // --- Input Handling & Simple Commands ---
            rule("Input 'search:Q' -> Search Memory", Struct(Type.INPUT, [Struct("search", [Variable("Q")])]), Struct("MemoryTool", [Atom("search"), Variable("Q")]), 15),
            rule("Input 'read:Path' -> Read File", Struct(Type.INPUT, [Struct("read", [Variable("Path")])]), Struct("FileSystemTool", [Atom("read_file"), Variable("Path")]), 15),
            rule("Input 'write:Path,Content' -> Write File", Struct(Type.INPUT, [Struct("write", [Variable("Path"), Variable("Content")])]), Struct("FileSystemTool", [Atom("write_file"), Variable("Path"), Variable("Content")]), 15),
            rule("Input 'list:Path' -> List Directory", Struct(Type.INPUT, [Struct("list", [Variable("Path")])]), Struct("FileSystemTool", [Atom("list_dir"), Variable("Path")]), 15),

            // --- Interaction (Examples - Adjust priorities/conditions) ---
            // rule("Confirm Generated Goal", Struct(Type.GOAL, [Variable("GoalContent")]), Struct("UserInteractionTool", [Atom("prompt"), Atom("CONFIRM_GOAL"), Struct("context", [Atom("goal:?GoalContent")])]), 5),
            // rule("Confirm Strategy Step", Struct(Type.STRATEGY, [Variable("StrategyContent")]), Struct("UserInteractionTool", [Atom("prompt"), Atom("CONFIRM_STRATEGY"), Struct("context", [Atom("strategy:?StrategyContent")])]), 0),
            // rule("Clarify Brief Input (Atom)", Struct(Type.INPUT, [Atom("?InputAtom")]), Struct("UserInteractionTool", [Atom("prompt"), Atom("CLARIFY_INPUT"), Struct("context", [Atom("input:?InputAtom")])]), -5),

            // --- Status & Control ---
            rule("Tool Result 'ok:*' -> Set Done", Atom("?Result"), Struct("CoreTool", [Atom("set_status"), Atom("self"), Atom(Status.DONE)]), -50), // Low priority catch-all for success atoms

            // --- Failure Handling ---
            // rule("Self-Correction: Simple Retry on Failure Log", Struct(Type.LOG, [Struct("failure", [Struct("thoughtId", [Variable("FailedId")]), Struct("content", [Variable("FailedContent")]), Struct("error", [Variable("ErrorMsg")])])]), Struct("CoreTool", [Atom("set_status"), Variable("FailedId"), Atom(Status.PENDING)]), 40), // Highest priority failure handler
            // rule("Self-Correction: LLM Suggest Recovery on Failure Log", Struct(Type.LOG, [Struct("failure", [Struct("thoughtId", [Variable("FailedId")]), Struct("content", [Variable("FailedContent")]), Struct("error", [Variable("ErrorMsg")])])]), Struct("LLMTool", [Atom("generate"), Atom("GENERATE_RECOVERY_STRATEGY"), Struct("context", [Atom("failedId:?FailedId"), Atom("failedContent:?FailedContent"), Atom("error:?ErrorMsg")])]), 39), // Next attempt
            rule("Ask User How to Handle Failure Log", Struct(Type.LOG, [Struct("failure", [Struct("thoughtId", [Variable("FailedId")]), Struct("content", [Variable("FailedContent")]), Struct("error", [Variable("ErrorMsg")])])]), Struct("UserInteractionTool", [Atom("prompt"), Atom("HANDLE_FAILURE"), Struct("context", [Atom("failedThoughtId:?FailedId"), Atom("content:?FailedContent"), Atom("error:?ErrorMsg")])]), 30), // Lower priority

            // --- Time & Maintenance ---
            rule("Initialize Periodic Tick on Startup", Struct(Type.SYSTEM, [Atom("startup")]), Struct("TimeTool", [Atom("periodic_tick"), Atom(TICK_INTERVAL_MS.toString())]), 200), // High priority on startup
            // Note: Actual checking for timed waits/stuck thoughts happens in the System loop, not via rules triggered by 'tick' for simplicity here.
            // Rules *could* match Struct("tick", ...) to perform periodic actions if needed.

            // --- Task Completion ---
            rule("Task Completion Log -> Summarize", Struct(Type.LOG, [Struct("task_completed", [Variable("GoalId")])]), Struct("ContextSummarizer", [Atom("summarize"), Variable("GoalId")]), 90),
            rule("Store Task Summary as Log", Struct(Type.ATOM, [Variable("Summary")]), Struct("CoreTool", [Atom("add_thought"), Atom(Type.LOG), Struct("task_summary", [Variable("Summary")]), Atom("self"), Atom("self")]), 80), // Triggered after ContextSummarizer returns Atom
            // rule("Task Completion Log -> Ask User for Next", Struct(Type.LOG, [Struct("task_completed", [Variable("GoalId")])]), Struct("UserInteractionTool", [Atom("prompt"), Atom("NEXT_TASK_PROMPT"), Struct("context", [Atom("taskId:?GoalId"), Atom("taskContent:?")])]), 100), // Needs GoalContent binding - complex rule needed
        ];

        rulesToAdd.forEach(r => this.rules.add(r));
        // Add initial system thought to trigger startup rules (like periodic tick)
        this.engine.addThought({
            id: generateId(), type: Type.SYSTEM, content: Atom("startup"), belief: Belief.DEFAULT, status: Status.PENDING,
            metadata: { created: new Date().toISOString(), provenance: 'system_init', priority: 300 } // High priority
        });
        log('info', `${this.rules.count()} default rules bootstrapped.`);
    }

    async initialize(): Promise<void> {
        log('info', 'Initializing FlowMind System...');
        // Load state from files (creates dirs if needed)
        await loadState(this.thoughts, this.rules, this.memory);
        // Add default rules if none were loaded
        this.addBootstrapRules();
        // Start the API server
        this.apiServer.start();
        log('info', chalk.green('FlowMind Initialized.') + chalk.yellow(" Paused. Use API 'run' command or REPL 'run' to start processing."));
    }

    startProcessingLoop(): void {
        if (this.isRunning) { log('warn', "Processing already running."); return; }
        log('info', chalk.green('Starting processing loop...'));
        this.isRunning = true;
        this.apiServer.broadcast({ type: 'status_update', payload: this.systemControl.getStatus() });

        // Clear any previous interval just in case
        if (this.workerIntervalId) clearInterval(this.workerIntervalId);

        this.workerIntervalId = setInterval(async () => {
            if (!this.isRunning) return; // Stop if paused externally

            try {
                // --- Maintenance Tasks (Run every interval) ---
                const now = Date.now();
                let maintenancePerformed = false;

                // 1. Check for timed waits to wake up
                this.thoughts.getAll().forEach(t => {
                    if (t.status === Status.WAITING && typeof t.metadata.waitingFor === 'object' && t.metadata.waitingFor.type === 'time' && t.metadata.waitingFor.timestamp <= now) {
                        log('debug', `Waking up thought ${shortId(t.id)} from timed wait.`);
                        t.status = Status.PENDING;
                        delete t.metadata.waitingFor;
                        this.thoughts.update(t);
                        maintenancePerformed = true;
                    }
                });

                // 2. Check for stuck thoughts (optional - can be noisy)
                // const stuck = this.thoughts.findStuckThoughts(STUCK_THRESHOLD_MS);
                // if (stuck.length > 0) {
                //     log('warn', `Found ${stuck.length} potentially stuck thoughts: ${stuck.map(t => shortId(t.id)).join(', ')}`);
                //     // TODO: Add logic to handle stuck thoughts (e.g., log, prompt user, mark failed)
                // }

                // --- Core Processing ---
                const processedCount = await this.engine.processBatch();

                // Save state if any thoughts were processed or maintenance occurred
                if (processedCount > 0 || maintenancePerformed) {
                    debouncedSaveState(this.thoughts, this.rules, this.memory);
                }
            } catch (error: any) {
                log('error', 'Error in processing loop:', error.message);
                this.pauseProcessingLoop(); // Pause on critical loop error
            }
        }, WORKER_INTERVAL);
    }

    pauseProcessingLoop(): void {
        if (!this.isRunning) { log('warn', "Processing already paused."); return; }
        log('warn', chalk.yellow('Pausing processing loop...'));
        this.isRunning = false;
        this.apiServer.broadcast({ type: 'status_update', payload: this.systemControl.getStatus() });
        if (this.workerIntervalId) { clearInterval(this.workerIntervalId); this.workerIntervalId = null; }
        // Ensure any pending debounced save is flushed on pause
        debouncedSaveState.flush();
        saveState(this.thoughts, this.rules, this.memory); // Save final state on pause
    }

    async shutdown(): Promise<void> {
        log('info', chalk.magenta('\nShutting down FlowMind System...'));
        this.pauseProcessingLoop(); // Pause loop and save state
        this.apiServer.stop(); // Stop API server
        debouncedSaveState.cancel(); // Cancel any pending debounced saves

        // Final synchronous save attempt
        try {
            const state: AppState = { thoughts: this.thoughts.toJSON(), rules: this.rules.toJSON() };
            fsSync.writeFileSync(STATE_FILE, JSON.stringify(state, null, 2));
            if (this.memory.isReady) {
                // FaissStore save is async, but we try anyway
                await this.memory.save().catch(err => log('error', "Error during final memory save:", err.message));
            }
            log('info', 'Final state saved.');
        } catch (error: any) {
            log('error', 'Error during final synchronous state save:', error.message);
        }

        log('info', chalk.green('FlowMind shutdown complete. Goodbye!'));
        await sleep(200); // Short delay for logs to flush
        process.exit(0);
    }
}

// --- REPL Client Utilities ---
const contentStr = (t: any): string => t?.content ? Terms.toString(Terms.jsonToTerm(t.content) ?? Atom("Invalid Content")) : '';

const formatThought = (t: Thought | any, detail = false): string => {
    const thoughtContent = contentStr(t);
    const detailLimit = detail ? 150 : 50;
    const statusMap: Record<Status, chalk.ChalkFunction> = {
        [Status.PENDING]: chalk.cyan, [Status.ACTIVE]: chalk.blueBright, [Status.WAITING]: chalk.yellow,
        [Status.DONE]: chalk.gray, [Status.FAILED]: chalk.red,
    };
    const statusColor = statusMap[t.status as Status] ?? chalk.magenta; // Default color for unknown status
    const errorIndicator = t.status === Status.FAILED ? chalk.red(' E') : '';
    const waitingIndicator = t.status === Status.WAITING ? chalk.yellow(' W') : '';
    const beliefScore = t.belief ? ` B:${t.belief.score?.().toFixed(2) ?? 'N/A'}` : ''; // Handle potential missing belief/score
    return `${chalk.magenta(shortId(t.id))}|${chalk.yellow(t.type)}|${statusColor(t.status ?? 'N/A')}${errorIndicator}${waitingIndicator}|${thoughtContent.substring(0, detailLimit)}${thoughtContent.length > detailLimit ? '...' : ''}${chalk.dim(beliefScore)}`;
};

const formatRule = (r: Rule | any): string => {
    const patternTerm = Terms.jsonToTerm(r.pattern);
    const actionTerm = Terms.jsonToTerm(r.action);
    const beliefScore = r.belief ? ` B:${r.belief.score?.().toFixed(2) ?? 'N/A'}` : '';
    return `${chalk.blue(shortId(r.id))}|P${r.metadata?.priority ?? 0}|${Terms.format(patternTerm, false)} -> ${Terms.format(actionTerm, false)} (${(r.metadata?.description ?? '').substring(0, 40)}...)${chalk.dim(beliefScore)}`;
};

const formatTaskSummary = (t: any): string => {
    const taskContent = contentStr(t);
    const prompts = t.pendingPrompts || [];
    const statusColorMap: Record<TaskStatus | string, chalk.ChalkFunction> = { 'RUNNING': chalk.green, 'PAUSED': chalk.yellow };
    const statusColor = statusColorMap[t.taskStatus] ?? chalk.magenta; // Default color
    const promptIndicator = prompts.length > 0 ? chalk.red(` [${prompts.length} Prompt(s)]`) : '';
    const mainStatusColor = t.status === Status.FAILED ? chalk.red : (t.status === Status.DONE ? chalk.gray : statusColor);
    return `${chalk.magenta(shortId(t.id))}|${chalk.yellow(t.type)}|${mainStatusColor(t.status)} (${statusColor(t.taskStatus)})|${taskContent.substring(0, 60)}${taskContent.length > 60 ? '...' : ''}${promptIndicator}`;
};

const formatTaskDetailItem = (t: any): string => {
    const taskContent = contentStr(t);
    const statusMap: Record<Status, chalk.ChalkFunction> = {
        [Status.PENDING]: chalk.cyan, [Status.ACTIVE]: chalk.blueBright, [Status.WAITING]: chalk.yellow,
        [Status.DONE]: chalk.gray, [Status.FAILED]: chalk.red,
    };
    const statusColor = statusMap[t.status as Status] ?? chalk.magenta;
    const errorIndicator = t.status === Status.FAILED ? chalk.red(' E') : '';
    const waitingIndicator = t.status === Status.WAITING ? chalk.yellow(' W') : '';
    return `  - ${chalk.magenta(shortId(t.id))}|${chalk.yellow(t.type)}|${statusColor(t.status)}${errorIndicator}${waitingIndicator}|${taskContent.substring(0, 100)}${taskContent.length > 100 ? '...' : ''}`;
};

// --- REPL Client ---
async function startReplClient(serverUrl: string) {
    console.log(chalk.cyan(`FlowMind REPL Client`));
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, prompt: chalk.blueBright('FM> ') });
    let ws: WebSocket | null = null;
    let requestCounter = 0;
    const pendingRequests = new Map<string, { resolve: (value: any) => void, reject: (reason?: any) => void, command: string }>();
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let shuttingDown = false;
    let serverStatus: { isRunning: boolean } = { isRunning: false }; // Track server status

    const updatePrompt = () => {
        const statusIndicator = serverStatus.isRunning ? chalk.green('▶') : chalk.yellow('❚❚');
        rl.setPrompt(`${statusIndicator} ${chalk.blueBright('FM> ')}`);
        rl.prompt();
    };

    const connect = () => {
        if (shuttingDown || (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING))) return;
        if (reconnectTimeout) clearTimeout(reconnectTimeout); reconnectTimeout = null;
        console.log(chalk.yellow(`Attempting connection to ${serverUrl}...`));
        ws = new WebSocket(serverUrl);

        ws.on('open', () => { console.log(chalk.green(`Connected to FlowMind server! Type 'help' for commands.`)); updatePrompt(); });
        ws.on('message', (data) => {
            const message: WsMessage | null = safeJsonParse(data.toString(), null);
            if (!message) { console.log(chalk.yellow('Received non-JSON message:', data.toString().substring(0, 100))); return; }
            const { type, payload, success, requestId, ...rest } = message as { type?: string; payload?: any; success?: boolean; requestId?: string; [key: string]: any };

            // Clear the current prompt line before printing output
            process.stdout.clearLine(0);
            process.stdout.cursorTo(0);

            if (requestId && pendingRequests.has(requestId)) {
                const reqInfo = pendingRequests.get(requestId)!; pendingRequests.delete(requestId);
                switch (type) {
                    case 'response': success ? reqInfo.resolve(payload) : reqInfo.reject(new Error(payload || `Command '${reqInfo.command}' failed.`)); break;
                    case 'error': reqInfo.reject(new Error(payload || `Server error for '${reqInfo.command}'.`)); break;
                    default: reqInfo.reject(new Error(`Unexpected message type '${type}' for request '${reqInfo.command}'.`)); break;
                }
            } else { // Broadcast messages
                switch (type) {
                    case 'error': console.error(chalk.red(`Server Error: ${payload}`)); break;
                    case 'snapshot':
                        console.log(chalk.magenta('--- Received State Snapshot ---'));
                        serverStatus = payload.status ?? { isRunning: false };
                        console.log(chalk.cyan(`Thoughts: ${payload.thoughts?.length ?? 0}, Rules: ${payload.rules?.length ?? 0}, Status: ${serverStatus.isRunning ? chalk.green('RUNNING') : chalk.yellow('PAUSED')}`));
                        break;
                    case 'thoughts_delta':
                        if (payload.changed.length > 0) console.log(chalk.greenBright(`Thoughts Updated:`), payload.changed.map((t: Thought) => formatThought(t)).join(', '));
                        if (payload.deleted.length > 0) console.log(chalk.redBright(`Thoughts Deleted: ${payload.deleted.map(shortId).join(', ')}`));
                        break;
                    case 'rules_delta':
                        if (payload.changed.length > 0) console.log(chalk.blueBright(`Rules Updated:`), payload.changed.map((r: Rule) => shortId(r.id)).join(', '));
                        if (payload.deleted.length > 0) console.log(chalk.yellowBright(`Rules Deleted: ${payload.deleted.map(shortId).join(', ')}`));
                        break;
                    case 'status_update':
                        serverStatus = payload ?? { isRunning: false };
                        console.log(chalk.magenta(`System Status: ${serverStatus.isRunning ? chalk.green('RUNNING') : chalk.yellow('PAUSED')}`));
                        break;
                    default: console.log(chalk.gray(`Server Message [${type || 'unknown'}]:`), payload ?? rest); break;
                }
            }
            // Re-display the prompt after handling the message
            updatePrompt();
        });        ws.on('close', (code, reason) => {
            if (shuttingDown) return;
            process.stdout.clearLine(0); process.stdout.cursorTo(0);
            console.log(chalk.red(`\nDisconnected (Code: ${code}, Reason: ${reason || 'N/A'}). Attempting reconnect...`));
            ws = null; serverStatus.isRunning = false; // Assume paused on disconnect
            pendingRequests.forEach(p => p.reject(new Error("Disconnected"))); pendingRequests.clear();
            if (!reconnectTimeout) reconnectTimeout = setTimeout(connect, RECONNECT_DELAY);
            updatePrompt();
        });
        ws.on('error', (error) => {
            if (shuttingDown) return;
            process.stdout.clearLine(0); process.stdout.cursorTo(0);
            console.error(chalk.red(`\nConnection Error: ${error.message}.`));
            ws?.close(); ws = null; serverStatus.isRunning = false;
            if (!reconnectTimeout) reconnectTimeout = setTimeout(connect, RECONNECT_DELAY);
            updatePrompt();
        });
    };

    const sendCommand = (command: string, payload: any): Promise<any> => {
        return new Promise((resolve, reject) => {
            if (!ws || ws.readyState !== WebSocket.OPEN) { reject(new Error("Not connected to server.")); return; }
            const requestId = `req-${requestCounter++}`;
            pendingRequests.set(requestId, { resolve, reject, command });
            ws.send(JSON.stringify({ command, payload, requestId }));
            // Timeout for request
            setTimeout(() => {
                if (pendingRequests.has(requestId)) {
                    pendingRequests.delete(requestId);
                    reject(new Error(`Request timeout for command '${command}'.`));
                    process.stdout.clearLine(0); process.stdout.cursorTo(0);
                    updatePrompt(); // Re-show prompt after timeout message
                }
            }, CLIENT_REQUEST_TIMEOUT);
        });
    };

    rl.on('line', async (line) => {
        const trimmed = line.trim();
        if (!trimmed) { updatePrompt(); return; } // Empty line, just show prompt
        if (trimmed === 'quit' || trimmed === 'exit') { rl.close(); return; }

        // Simple parsing: command is first word, rest is args. Handles quotes.
        const parts = trimmed.match(/(?:[^\s"]+|"[^"]*")+/g) ?? [];
        const command = parts[0]?.toLowerCase();
        const args = parts.slice(1).map(arg => arg.startsWith('"') && arg.endsWith('"') ? arg.slice(1, -1) : arg); // Unquote args

        if (!command) { updatePrompt(); return; }

        let payload: any = {}; let valid = true;
        try {
            // Construct payload based on command
            switch (command) {
                case 'add': payload = { text: args.join(' ') }; valid = args.length >= 1; break;
                case 'respond': payload = { promptId: args[0], text: args.slice(1).join(' ') }; valid = args.length >= 2; break;
                case 'info': case 'delete': case 'task': case 'pause_task': case 'run_task': payload = { idPrefix: args[0] }; valid = args.length === 1; break;
                case 'tag': case 'untag': payload = { idPrefix: args[0], tag: args[1] }; valid = args.length === 2; break;
                case 'search': payload = { query: args[0], k: args[1] ? parseInt(args[1], 10) : 5 }; valid = args.length >= 1 && (!args[1] || !isNaN(payload.k)); break;
                case 'run': case 'pause': case 'step': case 'save': case 'status': case 'thoughts': case 'rules': case 'tasks': case 'tools': case 'help': case 'ping': payload = {}; valid = args.length === 0; break;
                default: console.log(chalk.yellow(`Unknown command: ${command}. Type 'help'.`)); valid = false; break;
            }
            if (!valid) { console.log(chalk.yellow(`Invalid arguments for '${command}'. Type 'help'.`)); updatePrompt(); return; }

            // Send command and handle result/error
            const result = await sendCommand(command, payload);

            // --- Display formatted results ---
            if (command === 'help' && result?.commands) {
                console.log(chalk.cyan(result.description ?? 'Help:'));
                result.commands.forEach((c: { cmd: string, desc: string }) => console.log(`  ${chalk.blueBright(c.cmd.padEnd(25))}: ${c.desc}`));
            } else if (command === 'info' && result?.type === 'thought') {
                console.log(chalk.cyan('--- Thought Info ---')); console.log(formatThought(result.data, true)); console.log(chalk.gray(JSON.stringify(result.data.metadata, null, 2)));
            } else if (command === 'info' && result?.type === 'rule') {
                console.log(chalk.cyan('--- Rule Info ---')); console.log(formatRule(result.data)); console.log(chalk.gray(JSON.stringify(result.data.metadata, null, 2)));
            } else if (command === 'thoughts' && Array.isArray(result)) {
                console.log(chalk.cyan(`--- All Thoughts (${result.length}) ---`)); result.forEach((t: Thought) => console.log(formatThought(t)));
            } else if (command === 'rules' && Array.isArray(result)) {
                console.log(chalk.cyan(`--- All Rules (${result.length}) ---`)); result.forEach((r: Rule) => console.log(formatRule(r)));
            } else if (command === 'tasks' && Array.isArray(result)) {
                console.log(chalk.cyan(`--- Tasks (${result.length}) ---`)); result.forEach((t: any) => console.log(formatTaskSummary(t)));
            } else if (command === 'task' && result?.task) {
                console.log(chalk.cyan(`--- Task Details: ${formatTaskSummary(result.task)} ---`));
                const printSection = (title: string, items: any[] | undefined) => { if (items?.length) { console.log(chalk.blueBright(title + ':')); items.forEach((i: any) => console.log(formatTaskDetailItem(i))); } };
                printSection('Plan (Strategies)', result.plan); printSection('Outcomes', result.outcomes); printSection('Facts', result.facts); printSection('Prompts', result.prompts); printSection('Logs', result.logs); printSection('Other Thoughts', result.other);
            } else if (command === 'tools' && Array.isArray(result)) {
                console.log(chalk.cyan('--- Available Tools ---')); result.forEach((t: { name: string, description: string }) => console.log(`  ${chalk.blueBright(t.name.padEnd(20))}: ${t.description}`));
            } else if (command === 'search' && Array.isArray(result)) {
                console.log(chalk.cyan(`--- Memory Search Results (${result.length}) ---`));
                result.forEach((r: MemoryEntry, i: number) => console.log(`${i + 1}. [${shortId(r.metadata.sourceId)}] (Score: ${r.score?.toFixed(3) ?? 'N/A'}) ${r.content.substring(0, 100)}...`));
            } else if (result !== null && result !== undefined) { console.log(chalk.cyan(`Result:`), typeof result === 'object' ? JSON.stringify(result, null, 2) : result); }
            else { console.log(chalk.green('OK')); } // For commands that don't return specific payload

        } catch (error: any) { console.error(chalk.red(`Error:`), error.message); }
        finally { updatePrompt(); } // Show prompt again after command finishes
    });

    rl.on('close', () => {
        shuttingDown = true; console.log(chalk.yellow('\nREPL Client exiting.'));
        if (reconnectTimeout) clearTimeout(reconnectTimeout);
        if (ws && ws.readyState === WebSocket.OPEN) ws.close();
        setTimeout(() => process.exit(0), 100); // Allow time for close message
    });

    connect(); // Initial connection attempt
}

// --- Main Execution ---
async function main() {
    const mode = process.argv[2] || 'server'; // Default to server mode

    if (mode === 'server') {
        let system: System | null = null;
        const handleShutdown = async (signal: string) => {
            console.log(''); // Newline after ^C
            log('warn', `${signal} received. Shutting down gracefully...`);
            if (system) {
                // Prevent double shutdown calls
                process.off('SIGINT', handleShutdown);
                process.off('SIGTERM', handleShutdown);
                await system.shutdown();
            } else process.exit(0);
        };
        process.on('SIGINT', handleShutdown);
        process.on('SIGTERM', handleShutdown);

        try {
            system = new System();
            await system.initialize();
            // Server runs indefinitely until shutdown signal
        } catch (error: any) {
            log('error', "Critical initialization error:", error.message);
            console.error(error.stack);
            process.exit(1);
        }
    } else if (mode === 'client') {
        const serverUrl = process.argv[3] || `ws://localhost:${WS_PORT}`;
        await startReplClient(serverUrl);
        // REPL client runs until user quits
    } else {
        console.error(chalk.red(`Invalid mode: '${mode}'. Use 'server' or 'client [url]'`));
        process.exit(1);
    }
}

// Start the application
main().catch(error => {
    log('error', 'Unhandled error in main execution:', error.message);
    console.error(error.stack);
    process.exit(1);
});