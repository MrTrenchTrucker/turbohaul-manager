import { Component, ErrorInfo, ReactNode } from 'react';

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

/**
 * Top-level ErrorBoundary that catches any uncaught render error
 * and shows a visible fallback instead of blanking the entire page.
 *
 * Root cause: React 18 unmounts the whole tree on an uncaught
 * render throw when no ErrorBoundary is present — resulting in
 * a blank #root div (observed as a blank page when a null vram
 * value slipped through).
 */
export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Log to console for debugging — no external error reporting yet
    console.error('[ErrorBoundary] Uncaught render error:', error);
    console.error('[ErrorBoundary] Component stack:', info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-slate-950">
          <div className="max-w-md mx-auto p-6 rounded-lg border border-red-800 bg-slate-900">
            <h2 className="text-lg font-bold text-red-400 mb-3">
              Something went wrong
            </h2>
            <pre className="text-sm text-red-300 bg-slate-950 p-3 rounded overflow-auto max-h-64 mb-3">
              {this.state.error?.message ?? 'Unknown error'}
            </pre>
            <div className="text-xs text-slate-500">
              Refresh the page or check the browser console for details.
            </div>
            <button
              className="mt-3 px-3 py-1.5 text-sm bg-slate-800 text-slate-200 rounded hover:bg-slate-700"
              onClick={() => this.setState({ hasError: false, error: null })}
            >
              Retry
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
