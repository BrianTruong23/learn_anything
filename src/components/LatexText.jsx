import React from 'react';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';

/**
 * component that parses a string for LaTeX delimiters and renders them using KaTeX.
 * Supported delimiters:
 * - Inline: \( ... \)
 * - Block: \[ ... \]
 */
function LatexText({ text }) {
  if (!text) return null;

  // Split by block math first: \[ ... \] OR $$ ... $$
  const blockParts = text.split(/(\\\[.*?\\\]|\$\$[\s\S]*?\$\$)/s);

  return (
    <span>
      {blockParts.map((part, index) => {
        if ((part.startsWith('\\[') && part.endsWith('\\]')) || (part.startsWith('$$') && part.endsWith('$$'))) {
          // Remove delimiters and render BlockMath
          const math = part.startsWith('$$') ? part.slice(2, -2) : part.slice(2, -2);
          return <BlockMath key={index} math={math} />;
        } else {
          // Handle inline math within this part: \( ... \) OR $ ... $
          // Note: Regex for $...$ must be careful not to match $$...$$ (already handled) or escaped \$
          const inlineParts = part.split(/(\\\(.*?\\\)|(?<!\$)\$(?!\$)[\s\S]*?(?<!\$)\$(?!\$))/s);
          return (
            <span key={index}>
              {inlineParts.map((subPart, subIndex) => {
                if (subPart.startsWith('\\(') && subPart.endsWith('\\)')) {
                  const math = subPart.slice(2, -2);
                  return <InlineMath key={subIndex} math={math} />;
                } else if (subPart.startsWith('$') && subPart.endsWith('$')) {
                  const math = subPart.slice(1, -1);
                  return <InlineMath key={subIndex} math={math} />;
                } else {
                  return <span key={subIndex}>{subPart}</span>;
                }
              })}
            </span>
          );
        }
      })}
    </span>
  );
}

export default LatexText;
