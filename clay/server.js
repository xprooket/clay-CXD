#!/usr/bin/env node
/**
 * Clay MCP Server - JavaScript Edition
 * Robust stdio handling for Windows + Python bridge
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { 
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';

// Setup paths
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const CLAY_DIR = __dirname;
const PYTHON_SCRIPTS_DIR = path.join(CLAY_DIR, 'python_bridge');

// Logging utility
function log(message) {
  const timestamp = new Date().toISOString();
  console.error(`[CLAY-JS ${timestamp}] ${message}`);
}

// Python bridge utility
async function runPythonTool(toolName, args = []) {
  return new Promise((resolve, reject) => {
    log(`Executing Python tool: ${toolName} with args: ${JSON.stringify(args)}`);
    
    const scriptPath = path.join(PYTHON_SCRIPTS_DIR, `${toolName}.py`);
    
    // Check if script exists
    if (!fs.existsSync(scriptPath)) {
      reject(new Error(`Python script not found: ${scriptPath}`));
      return;
    }
    
    // Spawn Python process (use local venv Python)
    const isWindows = process.platform === 'win32';
    const pythonExecutable = isWindows 
      ? path.join(CLAY_DIR, 'venv', 'Scripts', 'python.exe')
      : path.join(CLAY_DIR, 'venv', 'bin', 'python');
    
    const pythonProcess = spawn(pythonExecutable, [scriptPath, ...args], {
      cwd: CLAY_DIR,
      stdio: ['pipe', 'pipe', 'pipe'],
      env: { 
        ...process.env, 
        PYTHONPATH: CLAY_DIR,
        // FORCE UTF-8 en Python subprocess
        PYTHONIOENCODING: 'utf-8',
        PYTHONUTF8: '1'  // Python 3.7+ UTF-8 mode
    }
});
    
    let stdout = '';
    let stderr = '';
    pythonProcess.stdout.setEncoding('utf8');
    pythonProcess.stderr.setEncoding('utf8');
    pythonProcess.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    pythonProcess.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          // Try to parse as JSON first, fallback to plain text
          const result = stdout.trim();
          if (result.startsWith('{') || result.startsWith('[')) {
            resolve(JSON.parse(result));
          } else {
            resolve(result);
          }
        } catch (parseError) {
          resolve(stdout.trim());
        }
      } else {
        log(`Python tool ${toolName} failed with code ${code}`);
        log(`STDERR: ${stderr}`);
        reject(new Error(`Python tool failed: ${stderr || 'Unknown error'}`));
      }
    });
    
    pythonProcess.on('error', (error) => {
      log(`Failed to start Python process: ${error.message}`);
      reject(error);
    });
    
    // Set timeout (longer for heavy tools like recall_cxd and context_tale)
    const timeout = (toolName === 'clay_recall_cxd' || toolName === 'clay_context_tale') ? 60000 : 30000;
    setTimeout(() => {
      pythonProcess.kill();
      reject(new Error(`Python tool ${toolName} timed out`));
    }, timeout);
  });
}

// Tool definitions
const CLAY_TOOLS = {
  remember: {
    name: 'remember',
    description: 'Store information in persistent memory',
    inputSchema: {
      type: 'object',
      properties: {
        content: {
          type: 'string',
          description: 'Content to remember'
        },
        memory_type: {
          type: 'string',
          description: 'Type of memory (interaction, reflection, synthetic, socratic)',
          default: 'interaction'
        }
      },
      required: ['content']
    }
  },
  recall: {
    name: 'recall',
    description: 'Search and retrieve relevant memories',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query for memories'
        },
        limit: {
          type: 'number',
          description: 'Maximum number of memories to return',
          default: 5
        },
        memory_type: {
          type: 'string',
          description: 'Filter by memory type (golden, reflection, milestone, etc.)',
          default: ''
        }
      },
      required: ['query']
    }
  },
  think_with_memory: {
    name: 'think_with_memory',
    description: 'Process input with full contextual memory',
    inputSchema: {
      type: 'object',
      properties: {
        input_text: {
          type: 'string',
          description: 'Input text to process with memory context'
        }
      },
      required: ['input_text']
    }
  },
  status: {
    name: 'status',
    description: 'Get Clay system status and statistics',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  classify_cxd: {
    name: 'classify_cxd',
    description: 'Classify text using CXD cognitive framework',
    inputSchema: {
      type: 'object',
      properties: {
        text: {
          type: 'string',
          description: 'Text to classify using CXD framework'
        }
      },
      required: ['text']
    }
  },
  recall_cxd: {
    name: 'recall_cxd',
    description: 'Search memories with CXD cognitive filtering',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Search query'
        },
        function_filter: {
          type: 'string',
          description: 'CXD function filter (CONTROL, CONTEXT, DATA, ALL)',
          default: 'ALL'
        },
        limit: {
          type: 'number',
          description: 'Maximum results',
          default: 5
        }
      },
      required: ['query']
    }
  },
  reflect: {
    name: 'reflect',
    description: 'Trigger offline reflection and pattern analysis',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  socratic_dialogue: {
    name: 'socratic_dialogue',
    description: 'Engage in Socratic self-questioning',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Topic or question for Socratic analysis'
        },
        depth: {
          type: 'number',
          description: 'Depth of Socratic questioning (1-5)',
          default: 3
        }
      },
      required: ['query']
    }
  },
  bootstrap_synthetic_memories: {
    name: 'bootstrap_synthetic_memories',
    description: 'Load foundational synthetic memories',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  analyze_memory_patterns: {
    name: 'analyze_memory_patterns',
    description: 'Analyze patterns in memory usage and content',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  debug_python: {
    name: 'debug_python',
    description: 'Debug which Python interpreter is being used',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  memory_transplant: {
    name: 'memory_transplant',
    description: 'Transplant memories to fix ID sequence holes',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  direct_insert: {
    name: 'direct_insert',
    description: 'Direct insert memory in specific ID slot',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  },
  update_memory_guided: {
    name: 'update_memory_guided',
    description: 'Update a memory with contextual guidance and Socratic questioning',
    inputSchema: {
      type: 'object',
      properties: {
        memory_id: {
          type: 'number',
          description: 'ID of the memory to update'
        }
      },
      required: ['memory_id']
    }
  },
  delete_memory_guided: {
    name: 'delete_memory_guided',
    description: 'Delete a memory with contextual analysis and confirmation',
    inputSchema: {
      type: 'object',
      properties: {
        memory_id: {
          type: 'number',
          description: 'ID of the memory to delete'
        },
        confirm: {
          type: 'boolean',
          description: 'Confirm deletion after analysis',
          default: false
        }
      },
      required: ['memory_id']
    }
  },
  context_tale: {
    name: 'context_tale',
    description: 'Generate fluid narratives from memory fragments - perfect for onboarding new Claude instances',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'What story to tell (e.g., "introduction", "project history")'
        },
        chunk_size: {
          type: 'number',
          description: 'Target tokens per chunk for small LLMs (default: 500)',
          default: 500
        },
        max_memories: {
          type: 'number',
          description: 'Maximum memories to include in narrative (default: 20)',
          default: 20
        },
        function_filter: {
          type: 'string',
          description: 'CXD function filter (CONTROL, CONTEXT, DATA, ALL)',
          default: 'ALL'
        },
        style: {
          type: 'string',
          description: 'Narrative style (auto, introduction, technical, philosophical, general)',
          default: 'auto'
        },
        chunk_id: {
          type: 'number',
          description: 'Return only specific chunk number (optional)'
        }
      },
      required: ['query']
    }
  }
};

// Create MCP server
const server = new Server(
  {
    name: 'clay-memory-js',
    version: '1.0.0',
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: Object.values(CLAY_TOOLS)
  };
});

// Call tool handler
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    log(`Tool called: ${name} with args: ${JSON.stringify(args)}`);
    
    switch (name) {
      case 'remember': {
        const { content, memory_type = 'interaction' } = args;
        const result = await runPythonTool('clay_remember', [content, memory_type]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'recall': {
        const { query, limit = 5, memory_type = '' } = args;
        const pythonArgs = [query, limit.toString()];
        if (memory_type) {
          pythonArgs.push('--type', memory_type);
        }
        const result = await runPythonTool('clay_recall', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'think_with_memory': {
        const { input_text } = args;
        const result = await runPythonTool('clay_think', [input_text]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'status': {
        const result = await runPythonTool('clay_status', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'classify_cxd': {
        const { text } = args;
        const result = await runPythonTool('clay_classify_cxd', [text]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'recall_cxd': {
        const { query, function_filter = 'ALL', limit = 5 } = args;
        const result = await runPythonTool('clay_recall_cxd', [query, function_filter, limit.toString()]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'reflect': {
        const result = await runPythonTool('clay_reflect', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'socratic_dialogue': {
        const { query, depth = 3 } = args;
        const result = await runPythonTool('clay_socratic', [query, depth.toString()]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'bootstrap_synthetic_memories': {
        const result = await runPythonTool('clay_bootstrap', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'analyze_memory_patterns': {
        const result = await runPythonTool('clay_analyze_patterns', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'debug_python': {
        const result = await runPythonTool('clay_debug_python', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'memory_transplant': {
        const result = await runPythonTool('clay_transplant', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'direct_insert': {
        const result = await runPythonTool('clay_direct_insert', []);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'update_memory_guided': {
        const { memory_id } = args;
        const result = await runPythonTool('clay_update_memory_guided', [memory_id.toString()]);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'delete_memory_guided': {
        const { memory_id, confirm = false } = args;
        const pythonArgs = [memory_id.toString()];
        if (confirm) {
          pythonArgs.push('--confirm');
        }
        const result = await runPythonTool('clay_delete_memory_guided', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      case 'context_tale': {
        const { 
          query, 
          chunk_size = 500, 
          max_memories = 20, 
          function_filter = 'ALL', 
          style = 'auto',
          chunk_id 
        } = args;
        
        const pythonArgs = [
          query,
          '--chunk-size', chunk_size.toString(),
          '--max-memories', max_memories.toString(),
          '--filter', function_filter,
          '--style', style
        ];
        
        if (chunk_id) {
          pythonArgs.push('--chunk', chunk_id.toString());
        }
        
        const result = await runPythonTool('clay_context_tale', pythonArgs);
        return { content: [{ type: 'text', text: result }] };
      }
      
      default:
        throw new Error(`Unknown tool: ${name}`);
    }
  } catch (error) {
    log(`Error in tool ${name}: ${error.message}`);
    return { 
      content: [{ 
        type: 'text', 
        text: `❌ Error en ${name}: ${error.message}` 
      }],
      isError: true
    };
  }
});

// === ERROR HANDLING ===

process.on('uncaughtException', (error) => {
  log(`Uncaught Exception: ${error.message}`);
  log(`Stack: ${error.stack}`);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  log(`Unhandled Rejection at: ${promise}`);
  log(`Reason: ${reason}`);
  process.exit(1);
});

// === MAIN EXECUTION ===

async function main() {
  log('Clay MCP Server (JavaScript Edition) starting...');
  
  // Check if Python bridge directory exists
  if (!fs.existsSync(PYTHON_SCRIPTS_DIR)) {
    log(`Creating Python bridge directory: ${PYTHON_SCRIPTS_DIR}`);
    fs.mkdirSync(PYTHON_SCRIPTS_DIR, { recursive: true });
  }
  
  // Setup stdio transport with robust error handling
  const transport = new StdioServerTransport();
  
  try {
    log('Connecting to stdio transport...');
    await server.connect(transport);
    log('✅ Clay MCP Server connected and ready!');
  } catch (error) {
    log(`Failed to connect server: ${error.message}`);
    process.exit(1);
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  log('Received SIGINT, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  log('Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

// Start the server
main().catch((error) => {
  log(`Fatal error in main: ${error.message}`);
  log(`Stack: ${error.stack}`);
  process.exit(1);
});
