import React from 'react';
import './SimpleSyntaxHighlighter.css';

const SimpleSyntaxHighlighter = ({ code, language = 'python' }) => {
  if (!code) return null;

  // Simple regex-based highlighting for Python
  const highlightCode = (code) => {
    // Escape HTML to prevent injection
    let safeCode = code.replace(/&/g, "&amp;")
                       .replace(/</g, "&lt;")
                       .replace(/>/g, "&gt;");

    // 1. Strings (single and double quotes)
    safeCode = safeCode.replace(/(['"])(?:(?=(\\?))\2.)*?\1/g, '<span class="token string">$&</span>');

    // 2. Comments
    safeCode = safeCode.replace(/(#.*)/g, '<span class="token comment">$1</span>');

    // 3. Keywords
    const keywords = [
      'import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif', 
      'for', 'while', 'try', 'except', 'with', 'pass', 'continue', 'break', 
      'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'lambda', 
      'global', 'nonlocal', 'assert', 'del', 'raise', 'yield', 'await', 'async'
    ];
    const keywordRegex = new RegExp(`\\b(${keywords.join('|')})\\b`, 'g');
    // We need to be careful not to highlight keywords inside strings or comments
    // But since we already wrapped strings and comments in spans, we can avoid them by checking if we are inside a tag?
    // A simpler approach for this lightweight highlighter:
    // We can't easily skip already replaced spans with simple regex replace chaining on the same string.
    // So we need a slightly more robust parsing or just accept some imperfections.
    // BETTER APPROACH: Tokenize and then reconstruct.
    
    // Let's stick to a simpler approach that works for most cases:
    // We will use a tokenizer function instead of sequential replaces on the whole string.
  };

  // Tokenizer approach
  const tokenize = (code) => {
    const tokens = [];
    const regex = /(".*?"|'.*?'|#.*|\b(?:import|from|as|def|class|return|if|else|elif|for|while|try|except|with|pass|continue|break|True|False|None|and|or|not|in|is|lambda|global|nonlocal|assert|del|raise|yield|await|async)\b|\b\d+\b|[a-zA-Z_]\w*|[(){}\[\],:;=+\-*/%&|^<>!~]|\s+)/g;
    
    let match;
    let lastIndex = 0;

    while ((match = regex.exec(code)) !== null) {
        const text = match[0];
        let type = 'text';

        if (text.startsWith('#')) {
            type = 'comment';
        } else if (text.startsWith('"') || text.startsWith("'")) {
            type = 'string';
        } else if (/^\d+$/.test(text)) {
            type = 'number';
        } else if ([
            'import', 'from', 'as', 'def', 'class', 'return', 'if', 'else', 'elif', 
            'for', 'while', 'try', 'except', 'with', 'pass', 'continue', 'break', 
            'True', 'False', 'None', 'and', 'or', 'not', 'in', 'is', 'lambda', 
            'global', 'nonlocal', 'assert', 'del', 'raise', 'yield', 'await', 'async'
        ].includes(text)) {
            type = 'keyword';
        } else if (['self', 'super', 'print', 'len', 'range', 'enumerate', 'zip', 'map', 'filter', 'list', 'dict', 'set', 'tuple', 'int', 'float', 'str', 'bool'].includes(text)) {
             type = 'builtin';
        } else if (/^[A-Z]/.test(text) && text !== 'True' && text !== 'False' && text !== 'None') {
            // Heuristic for class names
            type = 'class-name';
        } else if (text.match(/^[a-zA-Z_]\w*$/)) {
             // Check if it's a function call (followed by '(' in the original code? - hard to know here without lookahead)
             // For now just 'identifier'
             type = 'identifier';
        }

        tokens.push({ text, type });
    }
    
    return tokens;
  };

  const tokens = tokenize(code);

  return (
    <pre className="simple-syntax-highlighter">
      <code>
        {tokens.map((token, index) => (
          <span key={index} className={`token ${token.type}`}>
            {token.text}
          </span>
        ))}
      </code>
    </pre>
  );
};

export default SimpleSyntaxHighlighter;
