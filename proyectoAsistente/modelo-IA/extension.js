const vscode = require('vscode');
const http = require('http');

// Función auxiliar para consultar a la API de FastAPI local sin librerías externas
function fetchSuggestion(contextText) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify({ context: contextText });
        const req = http.request({
            hostname: '127.0.0.1',
            port: 8000,
            path: '/predict',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(data)
            }
        }, (res) => {
            let body = '';
            res.on('data', chunk => body += chunk);
            res.on('end', () => {
                try {
                    resolve(JSON.parse(body));
                } catch (e) {
                    reject(e);
                }
            });
        });
        
        req.on('error', reject);
        req.write(data);
        req.end();
    });
}

function activate(context) {
    console.log('Felicidades, tu extensión "Bryan AI" está activa.');

    // Registramos el proveedor de autocompletado "fantasma" (Inline Completion)
    const provider = vscode.languages.registerInlineCompletionItemProvider(
        { pattern: '**' }, // Se activa en cualquier archivo, idealmente archivos .c
        {
            async provideInlineCompletionItems(document, position) {
                const rangeBeforeCursor = new vscode.Range(new vscode.Position(0, 0), position);
                const fullTextBefore = document.getText(rangeBeforeCursor);
                const contextText = fullTextBefore.slice(-150);

                if (contextText.length === 0) return { items: [] };

                try {
                    // Consultamos al modelo de Machine Learning
                    const data = await fetchSuggestion(contextText);
                    
                    if (data && data.suggestion) {
                        return {
                            items: [{
                                insertText: data.suggestion,
                                range: new vscode.Range(position, position)
                            }]
                        };
                    }
                } catch (err) {
                    // Si el servidor de Python está apagado, simplemente no hace nada
                }
                return { items: [] };
            }
        }
    );
    

    context.subscriptions.push(provider);
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};