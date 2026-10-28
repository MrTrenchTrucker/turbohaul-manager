// Hand-written fake DOM for mounting a React 18 client tree under node:test.
//
// Zero dependencies, and no side effects on import: nothing is installed until
// a test calls installFakeDom(). The handle it returns puts every global back
// exactly as it was when restore() is called.
//
// This is not a browser. There is no layout, no CSS, no event dispatch and no
// real timers. Timers the code under test asks for are recorded and run only
// when the test calls them, so the test decides when a poll tick happens.
// Network is whatever fetch routes the caller supplies.
//
// Two things fail on purpose instead of quietly: getBoundingClientRect throws,
// because a zero rect would hide a size dependency. Listeners are stored and
// never dispatched, so listenerCounts() on the handle is how a test checks
// that nothing was left registered.
//
// The surface here is only what React 18 client rendering and the components
// under test actually touch. Anything else is absent on purpose, so a missing
// method fails loudly instead of doing nothing quietly.

const ELEMENT_NODE = 1;
const TEXT_NODE = 3;
const DOCUMENT_NODE = 9;
const HTML_NAMESPACE = 'http://www.w3.org/1999/xhtml';

export type Listener = (event: unknown) => void;
export type TimerCallback = (...args: unknown[]) => void;

export interface FakeTimer {
  readonly id: number;
  readonly fn: TimerCallback;
  readonly ms: number;
  readonly repeat: boolean;
}

export interface FakeTimers {
  setTimeout(fn: TimerCallback, ms?: number): number;
  clearTimeout(id: number | undefined): void;
  setInterval(fn: TimerCallback, ms?: number): number;
  clearInterval(id: number | undefined): void;
  pending(): FakeTimer[];
}

export function createFakeTimers(): FakeTimers {
  let nextId = 1;
  const active = new Map<number, FakeTimer>();
  const add = (fn: TimerCallback, ms: number | undefined, repeat: boolean): number => {
    const id = nextId;
    nextId += 1;
    active.set(id, { id, fn, ms: ms ?? 0, repeat });
    return id;
  };
  const remove = (id: number | undefined): void => {
    if (id !== undefined) active.delete(id);
  };
  return {
    setTimeout: (fn, ms) => add(fn, ms, false),
    clearTimeout: (id) => remove(id),
    setInterval: (fn, ms) => add(fn, ms, true),
    clearInterval: (id) => remove(id),
    pending: () => [...active.values()],
  };
}

export class FakeEventTarget {
  private readonly listeners = new Map<string, Set<Listener>>();

  addEventListener(type: string, listener: Listener): void {
    const set = this.listeners.get(type) ?? new Set<Listener>();
    set.add(listener);
    this.listeners.set(type, set);
  }

  removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener);
  }

  /** Listeners currently registered on this target, across every event type. */
  listenerCount(): number {
    let total = 0;
    for (const set of this.listeners.values()) total += set.size;
    return total;
  }
}

export abstract class FakeNode extends FakeEventTarget {
  abstract readonly nodeType: number;
  parentNode: FakeNode | null = null;
  ownerDocument: FakeDocument | null = null;
  readonly childNodes: FakeNode[] = [];

  get firstChild(): FakeNode | null {
    return this.childNodes[0] ?? null;
  }

  get lastChild(): FakeNode | null {
    return this.childNodes[this.childNodes.length - 1] ?? null;
  }

  get nextSibling(): FakeNode | null {
    const siblings = this.parentNode?.childNodes;
    if (!siblings) return null;
    const at = siblings.indexOf(this);
    return at >= 0 && at + 1 < siblings.length ? siblings[at + 1] : null;
  }

  get textContent(): string {
    return this.childNodes.map((child) => child.textContent).join('');
  }

  set textContent(value: string) {
    for (const child of this.childNodes.slice()) this.removeChild(child);
    if (value !== '') this.insertBefore(new FakeText(value, this.ownerDocument), null);
  }

  appendChild<T extends FakeNode>(child: T): T {
    return this.insertBefore(child, null);
  }

  insertBefore<T extends FakeNode>(child: T, before: FakeNode | null): T {
    child.parentNode?.removeChild(child);
    if (before === null) {
      this.childNodes.push(child);
    } else {
      const at = this.childNodes.indexOf(before);
      if (at < 0) throw new Error('fakeDom: insertBefore reference node is not a child of this node');
      this.childNodes.splice(at, 0, child);
    }
    child.parentNode = this;
    child.ownerDocument = this.nodeType === DOCUMENT_NODE ? (this as unknown as FakeDocument) : this.ownerDocument;
    return child;
  }

  removeChild<T extends FakeNode>(child: T): T {
    const at = this.childNodes.indexOf(child);
    if (at < 0) throw new Error('fakeDom: removeChild target is not a child of this node');
    this.childNodes.splice(at, 1);
    child.parentNode = null;
    return child;
  }
}

export class FakeText extends FakeNode {
  readonly nodeType = TEXT_NODE;
  nodeValue: string;

  constructor(text: string, ownerDocument: FakeDocument | null) {
    super();
    this.nodeValue = text;
    this.ownerDocument = ownerDocument;
  }

  get textContent(): string {
    return this.nodeValue;
  }

  set textContent(value: string) {
    this.nodeValue = value;
  }
}

export class FakeStyle {
  [property: string]: unknown;

  setProperty(property: string, value: string): void {
    this[property] = value;
  }

  removeProperty(property: string): void {
    delete this[property];
  }
}

export class FakeElement extends FakeNode {
  readonly nodeType = ELEMENT_NODE;
  readonly style = new FakeStyle();
  readonly attributes = new Map<string, string>();
  scrollTop = 0;
  scrollLeft = 0;
  scrollHeight = 0;
  clientHeight = 0;
  clientWidth = 0;

  constructor(
    readonly tagName: string,
    readonly namespaceURI: string | null,
    ownerDocument: FakeDocument | null,
  ) {
    super();
    this.ownerDocument = ownerDocument;
  }

  get nodeName(): string {
    return this.tagName.toUpperCase();
  }

  get className(): string {
    return this.attributes.get('class') ?? '';
  }

  set className(value: string) {
    this.setAttribute('class', value);
  }

  setAttribute(name: string, value: string | number | boolean): void {
    this.attributes.set(name, String(value));
  }

  getAttribute(name: string): string | null {
    return this.attributes.get(name) ?? null;
  }

  hasAttribute(name: string): boolean {
    return this.attributes.has(name);
  }

  removeAttribute(name: string): void {
    this.attributes.delete(name);
  }

  focus(): void {
    // Focus has no effect in this fake: nothing is ever the active element.
  }

  blur(): void {
    // See focus().
  }

  getBoundingClientRect(): never {
    // A zero rect would let a component that sizes itself from it run without
    // any error, so this refuses instead of pretending there is a layout.
    throw new Error('fakeDom: there is no layout engine, so getBoundingClientRect has no real rect to return');
  }
}

export class FakeIFrameElement extends FakeElement {}

export class FakeDocument extends FakeNode {
  readonly nodeType = DOCUMENT_NODE;
  readonly documentElement: FakeElement;
  readonly body: FakeElement;
  defaultView: FakeWindow | null = null;

  constructor() {
    super();
    this.documentElement = new FakeElement('html', HTML_NAMESPACE, this);
    this.body = new FakeElement('body', HTML_NAMESPACE, this);
    this.appendChild(this.documentElement);
    this.documentElement.appendChild(this.body);
  }

  get activeElement(): FakeElement | null {
    return null;
  }

  createElement(tagName: string): FakeElement {
    return new FakeElement(tagName.toLowerCase(), HTML_NAMESPACE, this);
  }

  createElementNS(namespace: string | null, qualifiedName: string): FakeElement {
    return new FakeElement(qualifiedName, namespace, this);
  }

  createTextNode(text: string): FakeText {
    return new FakeText(text, this);
  }
}

export interface FakeLocation {
  protocol: string;
  host: string;
  href: string;
}

export class FakeWindow extends FakeEventTarget {
  readonly location: FakeLocation = {
    protocol: 'http:',
    host: 'fake-dom.test',
    href: 'http://fake-dom.test/',
  };
  readonly HTMLIFrameElement = FakeIFrameElement;
  readonly document: FakeDocument;
  // A top-level window: react-dom compares window.top with window.self.
  readonly top: FakeWindow;
  readonly self: FakeWindow;
  private readonly timers: FakeTimers;

  constructor(document: FakeDocument, timers: FakeTimers) {
    super();
    this.document = document;
    this.timers = timers;
    this.top = this;
    this.self = this;
  }

  setTimeout(fn: TimerCallback, ms?: number): number {
    return this.timers.setTimeout(fn, ms);
  }

  clearTimeout(id: number | undefined): void {
    this.timers.clearTimeout(id);
  }

  setInterval(fn: TimerCallback, ms?: number): number {
    return this.timers.setInterval(fn, ms);
  }

  clearInterval(id: number | undefined): void {
    this.timers.clearInterval(id);
  }

  getSelection(): null {
    // No selection model here. React only asks when it restores focus, and
    // nothing in these tests is ever focused.
    return null;
  }
}

export interface FakeSocket {
  readonly url: string;
  closed: boolean;
  close(): void;
}

export interface FakeDomOptions {
  /** Answers every fetch() call. The url is the exact string the code asked for. */
  fetch: (url: string, init: unknown) => Promise<unknown>;
}

export interface FakeDomHandle {
  readonly window: FakeWindow;
  readonly document: FakeDocument;
  readonly timers: FakeTimers;
  /** Every url passed to the fetch stub, in call order. */
  readonly fetchUrls: string[];
  /** Every WebSocket the code opened. None of them ever receives a message. */
  readonly sockets: FakeSocket[];
  /** Listeners registered on the fake window and on the fake document right now. */
  listenerCounts(): { readonly window: number; readonly document: number };
  /** Puts every global back the way it was before installFakeDom ran. */
  restore(): void;
}

const INSTALLED_GLOBALS = [
  'window',
  'document',
  'navigator',
  'fetch',
  'WebSocket',
  'setTimeout',
  'clearTimeout',
  'setInterval',
  'clearInterval',
  'IS_REACT_ACT_ENVIRONMENT',
] as const;

export function installFakeDom(options: FakeDomOptions): FakeDomHandle {
  const globals = globalThis as unknown as Record<string, unknown>;
  const saved = new Map<string, PropertyDescriptor | undefined>();
  for (const name of INSTALLED_GLOBALS) {
    saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
  }

  const timers = createFakeTimers();
  const document = new FakeDocument();
  const window = new FakeWindow(document, timers);
  document.defaultView = window;
  const fetchUrls: string[] = [];
  const sockets: FakeSocket[] = [];

  class FakeWebSocket implements FakeSocket {
    closed = false;
    onopen: unknown = null;
    onmessage: unknown = null;
    onclose: unknown = null;
    onerror: unknown = null;

    constructor(readonly url: string) {
      sockets.push(this);
    }

    close(): void {
      this.closed = true;
    }
  }

  const values: Record<string, unknown> = {
    window,
    document,
    // react-dom reads navigator.userAgent once at load, for a devtools hint.
    // An empty agent string keeps that hint silent.
    navigator: { userAgent: '' },
    fetch: (input: unknown, init: unknown) => {
      const url = String(input);
      fetchUrls.push(url);
      return options.fetch(url, init);
    },
    WebSocket: FakeWebSocket,
    setTimeout: timers.setTimeout,
    clearTimeout: timers.clearTimeout,
    setInterval: timers.setInterval,
    clearInterval: timers.clearInterval,
    IS_REACT_ACT_ENVIRONMENT: true,
  };
  for (const name of INSTALLED_GLOBALS) {
    Object.defineProperty(globalThis, name, {
      value: values[name],
      writable: true,
      configurable: true,
      enumerable: false,
    });
  }

  return {
    window,
    document,
    timers,
    fetchUrls,
    sockets,
    listenerCounts() {
      return { window: window.listenerCount(), document: document.listenerCount() };
    },
    restore(): void {
      for (const name of INSTALLED_GLOBALS) {
        const descriptor = saved.get(name);
        if (descriptor) {
          Object.defineProperty(globalThis, name, descriptor);
        } else {
          delete globals[name];
        }
      }
    },
  };
}

/** Pre-order walk: every element under root for which the predicate holds. */
export function findElements(root: FakeNode, predicate: (element: FakeElement) => boolean): FakeElement[] {
  const found: FakeElement[] = [];
  const visit = (node: FakeNode): void => {
    if (node instanceof FakeElement && predicate(node)) found.push(node);
    for (const child of node.childNodes) visit(child);
  };
  visit(root);
  return found;
}

/** Every element under root whose attribute equals value. */
export function findAllByAttribute(root: FakeNode, name: string, value: string): FakeElement[] {
  return findElements(root, (element) => element.getAttribute(name) === value);
}
