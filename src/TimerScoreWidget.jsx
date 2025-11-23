import React from 'react';
import './TimerScoreWidget.css';

function TimerScoreWidget({ seconds, score }) {
  const formatTime = (totalSeconds) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  };

  return (
    <div className="timer-score-widget">
      <div className="widget-row">
        <span className="widget-label">Time on platform:</span>
        <span className="widget-value">{formatTime(seconds)}</span>
      </div>
      <div className="widget-row">
        <span className="widget-label">Learning Points:</span>
        <span className="widget-value">{score}</span>
      </div>
    </div>
  );
}

export default TimerScoreWidget;
