import React from 'react';
import { BlockMath, InlineMath } from 'react-katex';
import 'katex/dist/katex.min.css';
import './MathEquation.css';

function MathEquation({ equation }) {
  if (!equation) return null;

  return (
    <div className="math-equation-block">
      {/* Main equation */}
      <div className="equation-display">
        <BlockMath math={equation.latex} />
      </div>

      {/* Term-by-term breakdown */}
      {equation.terms && Object.keys(equation.terms).length > 0 && (
        <div className="equation-terms">
          <strong>Terms:</strong>
          <ul className="terms-list">
            {Object.entries(equation.terms).map(([symbol, meaning]) => (
              <li key={symbol}>
                <InlineMath math={symbol} /> = {meaning}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Intuitive explanation */}
      {equation.intuition && (
        <div className="equation-intuition">
          <strong>💡 Intuition:</strong> {equation.intuition}
        </div>
      )}
    </div>
  );
}

export default MathEquation;
