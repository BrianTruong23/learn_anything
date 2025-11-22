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

  // Split by block math first: \[ ... \]
  const blockParts = text.split(/(\\\[.*?\\\])/s);

  return (
    <span>
      {blockParts.map((part, index) => {
        if (part.startsWith('\\[') && part.endsWith('\\]')) {
          // Remove delimiters and render BlockMath
          const math = part.slice(2, -2);
          return <BlockMath key={index} math={math} />;
        } else {
          // Handle inline math within this part: \( ... \)
          const inlineParts = part.split(/(\\\(.*?\\\))/s);
          return (
            <span key={index}>
              {inlineParts.map((subPart, subIndex) => {
                if (subPart.startsWith('\\(') && subPart.endsWith('\\)')) {
                  // Remove delimiters and render InlineMath
                  const math = subPart.slice(2, -2);
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
