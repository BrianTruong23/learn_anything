const fs = require('fs');
const path = require('path');

const sectionName = process.argv[2];

if (!sectionName) {
  console.error('Please provide a section name (e.g., CNN)');
  process.exit(1);
}

const sectionId = sectionName.toLowerCase().replace(/\s+/g, '_');
const sectionTitle = sectionName;
const conceptsVarName = `${sectionId}Concepts`;

const rootDir = path.resolve(__dirname, '..');
const srcDir = path.join(rootDir, 'src');
const dataDir = path.join(srcDir, 'data');
const componentsDir = path.join(srcDir, 'components');

// 1. Update conceptsData.js
const conceptsDataPath = path.join(dataDir, 'conceptsData.js');
let conceptsData = fs.readFileSync(conceptsDataPath, 'utf8');

const newConceptsData = `
export const ${conceptsVarName} = [
  {
    id: "${sectionId}_0",
    title: "Big Picture: ${sectionTitle}",
    architectureImage: "PLACEHOLDER", // Placeholder
    subConcepts: [
      { id: "${sectionId}_sub_0_1", label: "Concept 1" },
      { id: "${sectionId}_sub_0_2", label: "Concept 2" },
      { id: "${sectionId}_sub_0_3", label: "Concept 3" }
    ],
    explanations: {
      beginner: {
        motivation: "Motivation for ${sectionTitle}...",
        definition: "Definition of ${sectionTitle}...",
        toyExample: {
          description: "Toy example for ${sectionTitle}:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      },
      intermediate: {
        motivation: "Intermediate motivation...",
        definition: "Intermediate definition...",
        toyExample: {
          description: "Intermediate example:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      },
      advanced: {
        motivation: "Advanced motivation...",
        definition: "Advanced definition...",
        toyExample: {
          description: "Advanced example:",
          steps: ["Step 1", "Step 2", "Step 3"]
        }
      }
    }
  },
  {
    id: "${sectionId}_1",
    title: "Core Concept 1",
    subConcepts: [],
    explanations: {
      beginner: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      intermediate: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      advanced: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } }
    }
  },
  {
    id: "${sectionId}_2",
    title: "Core Concept 2",
    subConcepts: [],
    explanations: {
      beginner: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      intermediate: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } },
      advanced: { motivation: "...", definition: "...", toyExample: { description: "...", steps: [] } }
    }
  }
];
`;

conceptsData += newConceptsData;
fs.writeFileSync(conceptsDataPath, conceptsData);
console.log(`Updated conceptsData.js with ${conceptsVarName}`);

// 2. Update ConceptExplorer.jsx
const explorerPath = path.join(componentsDir, 'ConceptExplorer.jsx');
let explorerContent = fs.readFileSync(explorerPath, 'utf8');

const newCard = `
        {/* ${sectionTitle} Card */}
        <div className="concept-card-item primary-card">
          <div className="card-badge intermediate">Intermediate</div>
          <h2 className="card-title">${sectionTitle} Architecture</h2>
          <p className="card-description">
            Explore the fundamental concepts of ${sectionTitle}.
          </p>
          <button 
            className="card-cta-btn"
            onClick={() => onSelectConcept('${sectionId}')}
          >
            Start ${sectionTitle} Journey
          </button>
        </div>`;

// Insert using the marker
const insertionMarker = '{/* INSERT_NEW_CARD_HERE */}';
if (explorerContent.includes(insertionMarker)) {
  explorerContent = explorerContent.replace(insertionMarker, `${newCard}\n\n        ${insertionMarker}`);
  fs.writeFileSync(explorerPath, explorerContent);
  console.log(`Updated ConceptExplorer.jsx with ${sectionTitle} card`);
} else {
  console.error('Could not find insertion marker in ConceptExplorer.jsx');
}

// 3. Update App.jsx
const appPath = path.join(srcDir, 'App.jsx');
let appContent = fs.readFileSync(appPath, 'utf8');

// Helper function for regex replacement
function injectCondition(content, regex, injection) {
  return content.replace(regex, (match, p1) => `${p1}${injection}`);
}

// Import
// Matches: import { concepts, ... } from './data/conceptsData.js';
const importRegex = /(import \{ concepts.*?)( \} from '\.\/data\/conceptsData\.js';)/;
appContent = appContent.replace(importRegex, `$1, ${conceptsVarName}$2`);

// onSelectConcept
// Matches: if (conceptId === 'transformer' ... ) {
const selectRegex = /(if \(conceptId === 'transformer'.*?)(\) \{)/;
appContent = appContent.replace(selectRegex, `$1 || conceptId === '${sectionId}'$2`);

// Header Title
// Matches: <h1>{currentView === ...
const headerRegex = /(<h1>\{currentView === )/;
appContent = appContent.replace(headerRegex, `$1'${sectionId}' ? '${sectionTitle} Architecture' : currentView === `);

// ConceptSequence concepts prop
// Matches: concepts={currentView === ...
const sequenceRegex = /(concepts=\{currentView === )/;
appContent = appContent.replace(sequenceRegex, `$1'${sectionId}' ? ${conceptsVarName} : currentView === `);

// Selected Concepts Logic
// Matches: if (currentView === '...
// We need to prepend our condition to the first if
const selectedConceptsRegex = /(if \(currentView === ')/;
// This one is tricky because there are multiple ifs. We want the one inside useEffect.
// Let's search for the useEffect block start or just rely on the first one being the right one?
// The first one is in onSelectConcept (which we handled above with selectRegex).
// The second one is in useEffect.
// Let's be more specific.
const selectedConceptsBlockRegex = /(useEffect\(\(\) => \{\s+)(if \(currentView === )/;
appContent = appContent.replace(selectedConceptsBlockRegex, `$1if (currentView === '${sectionId}') {\n      setSelectedConcepts(${conceptsVarName}.map(c => c.id));\n    } else $2`);

// Sources Logic
// Matches: {(currentView === ...
const sourcesRegex = /(\{\(currentView === )/;
appContent = appContent.replace(sourcesRegex, `$1'${sectionId}' ? [] : currentView === `);


fs.writeFileSync(appPath, appContent);
console.log(`Updated App.jsx to handle ${sectionId} view`);

// 4. Update quizQuestionsData.js
const quizPath = path.join(dataDir, 'quizQuestionsData.js');
let quizContent = fs.readFileSync(quizPath, 'utf8');

const newQuestions = `
  // ${sectionTitle} Questions
  {
    id: "${sectionId}_q1",
    type: "mcq",
    conceptTag: "${sectionId}_0",
    question: "What is the primary goal of ${sectionTitle}?",
    options: [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    correctOptionIndex: 0,
    explanation: "Placeholder explanation for ${sectionTitle}."
  },
`;

// Append before end of array
quizContent = quizContent.replace('];', `${newQuestions}\n];`);
fs.writeFileSync(quizPath, quizContent);
console.log(`Updated quizQuestionsData.js with placeholder questions`);

console.log(`\nSuccessfully created section "${sectionTitle}"!`);
