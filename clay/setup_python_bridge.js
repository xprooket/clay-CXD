#!/usr/bin/env node
/**
 * Clay Setup Script
 * Configura el bridge Python y verifica dependencias
 */

import fs from 'fs';
import path from 'path';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Colores para console
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logStep(step, message) {
  log(`[${step}] ${message}`, 'cyan');
}

function logSuccess(message) {
  log(`✅ ${message}`, 'green');
}

function logError(message) {
  log(`❌ ${message}`, 'red');
}

function logWarning(message) {
  log(`⚠️  ${message}`, 'yellow');
}

async function runCommand(command, args = [], options = {}) {
  return new Promise((resolve, reject) => {
    const process = spawn(command, args, {
      stdio: ['inherit', 'pipe', 'pipe'],
      ...options
    });
    
    let stdout = '';
    let stderr = '';
    
    process.stdout.on('data', (data) => {
      stdout += data.toString();
    });
    
    process.stderr.on('data', (data) => {
      stderr += data.toString();
    });
    
    process.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout, stderr });
      } else {
        reject(new Error(`Command failed with code ${code}: ${stderr}`));
      }
    });
  });
}

async function checkPython() {
  logStep('PYTHON', 'Verificando instalación de Python...');
  
  try {
    const result = await runCommand('python', ['--version']);
    const version = result.stdout.trim() || result.stderr.trim();
    logSuccess(`Python encontrado: ${version}`);
    
    // Verificar sqlite3
    await runCommand('python', ['-c', 'import sqlite3; print("SQLite3 OK")']);
    logSuccess('SQLite3 disponible');
    
    return true;
  } catch (error) {
    logError('Python no encontrado o SQLite3 no disponible');
    logError('Instala Python 3.8+ con SQLite3');
    return false;
  }
}

async function checkClayStructure() {
  logStep('CLAY', 'Verificando estructura de Clay...');
  
  const requiredFiles = [
    'clay/memory.py',
    'clay/assistant.py',
    'clay/__init__.py'
  ];
  
  const requiredDirs = [
    'clay'
  ];
  
  let allGood = true;
  
  // Verificar directorios
  for (const dir of requiredDirs) {
    const dirPath = path.join(__dirname, dir);
    if (!fs.existsSync(dirPath)) {
      logError(`Directorio faltante: ${dir}`);
      allGood = false;
    } else {
      logSuccess(`Directorio encontrado: ${dir}`);
    }
  }
  
  // Verificar archivos
  for (const file of requiredFiles) {
    const filePath = path.join(__dirname, file);
    if (!fs.existsSync(filePath)) {
      logWarning(`Archivo faltante: ${file}`);
      // No marcar como error crítico, puede que tenga nombres diferentes
    } else {
      logSuccess(`Archivo encontrado: ${file}`);
    }
  }
  
  return allGood;
}

async function createPythonBridge() {
  logStep('BRIDGE', 'Configurando Python bridge...');
  
  const bridgeDir = path.join(__dirname, 'python_bridge');
  
  // Crear directorio si no existe
  if (!fs.existsSync(bridgeDir)) {
    fs.mkdirSync(bridgeDir, { recursive: true });
    logSuccess('Directorio python_bridge creado');
  } else {
    logSuccess('Directorio python_bridge ya existe');
  }
  
  // Lista de scripts que deberían existir
  const expectedScripts = [
    'clay_remember.py',
    'clay_recall.py',
    'clay_think.py',
    'clay_status.py',
    'clay_classify_cxd.py',
    'clay_reflect.py',
    'clay_socratic.py',
    'clay_bootstrap.py',
    'clay_recall_cxd.py',
    'clay_analyze_patterns.py'
  ];
  
  let scriptsFound = 0;
  for (const script of expectedScripts) {
    const scriptPath = path.join(bridgeDir, script);
    if (fs.existsSync(scriptPath)) {
      scriptsFound++;
      logSuccess(`Script encontrado: ${script}`);
    } else {
      logWarning(`Script faltante: ${script}`);
    }
  }
  
  log(`\nScripts encontrados: ${scriptsFound}/${expectedScripts.length}`, 'cyan');
  
  if (scriptsFound < expectedScripts.length) {
    logWarning('Algunos scripts bridge faltan. Cópialos manualmente al directorio python_bridge/');
  }
  
  return true;
}

async function testBasicFunctionality() {
  logStep('TEST', 'Probando funcionalidad básica...');
  
  try {
    // Test Python imports
    const testCode = `
import sys
import os
sys.path.insert(0, '.')
try:
    from clay.memory import Memory, MemoryStore
    print("Clay imports: OK")
except ImportError as e:
    print(f"Clay imports: ERROR - {e}")
`;
    
    fs.writeFileSync('test_clay_imports.py', testCode);
    
    const result = await runCommand('python', ['test_clay_imports.py']);
    
    if (result.stdout.includes('OK')) {
      logSuccess('Imports de Clay funcionando');
    } else {
      logError('Problemas con imports de Clay');
      console.log(result.stdout);
      console.log(result.stderr);
    }
    
    // Limpiar archivo de test
    fs.unlinkSync('test_clay_imports.py');
    
  } catch (error) {
    logError('Error en test básico de funcionalidad');
    logError(error.message);
  }
}

async function checkCXDDependencies() {
  logStep('CXD', 'Verificando dependencias de CXD...');
  
  // Verificar si existe el directorio core con CXD
  const cxdPath = path.join(__dirname, 'core');
  if (fs.existsSync(cxdPath)) {
    logSuccess('Directorio core/ encontrado');
    
    const cxdFiles = [
      'cxd_optimized_architecture.py',
      'cxd_standalone_architecture.py',
      'cxd_meta_classifier.py'
    ];
    
    let cxdFilesFound = 0;
    for (const file of cxdFiles) {
      if (fs.existsSync(path.join(cxdPath, file))) {
        cxdFilesFound++;
        logSuccess(`CXD file: ${file}`);
      }
    }
    
    if (cxdFilesFound > 0) {
      logSuccess(`CXD Classifier disponible (${cxdFilesFound} archivos)`);
    } else {
      logWarning('Archivos CXD no encontrados en core/');
    }
  } else {
    logWarning('Directorio core/ no encontrado - CXD no disponible');
    logWarning('Copia los archivos CXD al directorio core/ si los necesitas');
  }
}

function generateStartupInstructions() {
  logStep('DOCS', 'Generando instrucciones de inicio...');
  
  const instructions = `
# Clay MCP Server - Instrucciones de Inicio

## Para usar con Claude Desktop:

1. **Instalar dependencias de Node.js:**
   \`\`\`bash
   npm install
   \`\`\`

2. **Verificar Python:**
   \`\`\`bash
   npm run check-python
   \`\`\`

3. **Configurar en Claude Desktop:**
   Edita tu archivo de configuración de Claude y añade:
   
   \`\`\`json
   {
     "mcpServers": {
       "clay-memory": {
         "command": "node",
         "args": ["server.js"],
         "cwd": "${__dirname}"
       }
     }
   }
   \`\`\`

4. **Reiniciar Claude Desktop**

## Para desarrollo:

\`\`\`bash
npm run dev    # Inicia con debugging
npm start      # Inicia normal
\`\`\`

## Herramientas disponibles:

- \`remember\` - Guardar memorias
- \`recall\` - Buscar memorias
- \`think_with_memory\` - Procesar con contexto
- \`status\` - Estado del sistema
- \`classify_cxd\` - Análisis cognitivo CXD

¡Clay está listo para darte memoria persistente! 🧠
`;

  fs.writeFileSync('STARTUP_INSTRUCTIONS.md', instructions);
  logSuccess('Instrucciones guardadas en STARTUP_INSTRUCTIONS.md');
}

async function main() {
  log('\n🚀 CLAY MCP SERVER SETUP', 'bright');
  log('================================\n', 'bright');
  
  let setupOk = true;
  
  // Verificar Python
  if (!(await checkPython())) {
    setupOk = false;
  }
  
  console.log('');
  
  // Verificar estructura Clay
  if (!(await checkClayStructure())) {
    setupOk = false;
  }
  
  console.log('');
  
  // Configurar bridge
  await createPythonBridge();
  
  console.log('');
  
  // Verificar CXD
  await checkCXDDependencies();
  
  console.log('');
  
  // Test básico
  await testBasicFunctionality();
  
  console.log('');
  
  // Generar instrucciones
  generateStartupInstructions();
  
  console.log('');
  
  // Resumen final
  if (setupOk) {
    logSuccess('✨ Setup completado exitosamente!');
    log('\n📖 Lee STARTUP_INSTRUCTIONS.md para continuar', 'yellow');
  } else {
    logWarning('⚠️  Setup completado con advertencias');
    log('Revisa los errores arriba y corrige antes de continuar', 'yellow');
  }
  
  log('\n🎯 Próximo paso: npm start\n', 'cyan');
}

main().catch((error) => {
  logError(`Error fatal en setup: ${error.message}`);
  process.exit(1);
});
