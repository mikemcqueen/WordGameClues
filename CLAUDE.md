# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Word-game clue manager for nested word puzzles. Given word pairs (e.g., "blue", "origin"), the system finds solutions (e.g., "Jeff Bezos"), which become clues for deeper puzzle levels. The system manages clue databases, generates/validates word combinations, and uses GPU acceleration for filtering large candidate sets.

## Build Commands

```bash
# Install dependencies
npm install

# Build native module (requires CUDA toolkit)
node-gyp build

# Compile TypeScript (output to src/dist/)
cd src && tsc
```

## Architecture

### Entry Points
- **`src/tools/clues.js`** - Main CLI tool for clue operations (combo generation, validation, testing)
- **`src/tools/cm.ts`** - Modular "combo-maker" CLI with subcommands (pairs, solutions, populate, remain, retire, lines)

### Directory Structure
```
src/
├── tools/          # CLI entry points and utilities
├── modules/        # TypeScript/JavaScript business logic
│   └── test/       # Mocha tests (test-*.js)
├── cm/             # Subcommand modules for cm.ts
├── types/          # TypeScript type definitions
├── native-modules/ # C++/CUDA native code
│   └── experiment/ # Main native addon (→ build/experiment.node)
└── dist/           # Compiled TypeScript output
```

### Native Module (C++/CUDA)

The native module (`build/experiment.node`) handles performance-critical operations:
- **combo-maker.cpp** - Generate candidate word combinations
- **clue-manager.cpp** - Manage clues and sources in memory
- **components.cpp** - Validate source compatibility
- **filter.cu, merge.cu, or-filter.cu** - CUDA kernels for GPU-accelerated filtering

Build requires: CUDA toolkit in `/usr/local/cuda`, C++23 compiler

### Data Flow
```
CLI (clues.js/cm.ts) → TypeScript modules → Native C++ → CUDA kernels
```

### Key Types
- **NameCount** - `{ name: string, count: number }` - clue with letter count
- **Clue** - `{ name, src, note?, ignore?, synonym?, homonym? }` - full clue record

### Data Files
Clue data lives in `data/` as JSON files, organized by clue type and word count.

## Common CLI Usage

```bash
cd src/tools

# show components of a clue
# also used for testing memory-usage refactoring currently in progress
node clues -pf.72 -t constitution -m 18 --memory
```
