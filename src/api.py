"""
Servidor HTTP leve que serve:
  GET /metrics    → Métricas Prometheus
  GET /photos/{name} → Fotos de infrações
  GET /clips/{name}  → Clips de vídeo
  GET /health     → Health check
  GET /status     → Status JSON
"""
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread

from src.metrics import get_metrics, get_metrics_content_type

logger = logging.getLogger("traffic-monitor.api")


class TrafficHTTPHandler(BaseHTTPRequestHandler):
    """Handler HTTP com rotas do traffic monitor."""
    
    # Injetado pelo servidor
    data_dir = Path(".")
    
    def do_GET(self):
        path = self.path.split("?")[0]
        
        if path == "/metrics":
            self._serve_metrics()
        elif path == "/health":
            self._serve_json({"status": "ok"}, 200)
        elif path == "/status":
            self._serve_status()
        elif path.startswith("/photos/"):
            self._serve_file(self.data_dir / "clips", path.replace("/photos/", ""), "image/jpeg")
        elif path.startswith("/clips/"):
            self._serve_file(self.data_dir / "clips", path.replace("/clips/", ""), "video/mp4")
        else:
            self._serve_json({"error": "not found", "endpoints": [
                "/metrics", "/health", "/status", "/photos/{name}", "/clips/{name}"
            ]}, 404)
    
    def _serve_metrics(self):
        body = get_metrics()
        self.send_response(200)
        self.send_header("Content-Type", get_metrics_content_type())
        self.end_headers()
        self.wfile.write(body)
    
    def _serve_json(self, data: dict, code: int = 200):
        import json
        body = json.dumps(data, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)
    
    def _serve_status(self):
        from src.metrics import get_status_dict
        status = get_status_dict()
        status["data_dir"] = str(self.data_dir)
        self._serve_json(status)
    
    def _serve_file(self, base_dir: Path, filename: str, content_type: str):
        # Prevenir path traversal
        safe_name = Path(filename).name
        filepath = base_dir / safe_name
        
        if not filepath.exists():
            self._serve_json({"error": f"file not found: {safe_name}"}, 404)
            return
        
        try:
            file_size = filepath.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except Exception as e:
            logger.error(f"Erro ao servir arquivo {filepath}: {e}")
            self._serve_json({"error": "internal error"}, 500)
    
    def log_message(self, format, *args):
        # args: (request_line, code_string, size_string)
        try:
            code = int(args[1]) if len(args) >= 2 else 0
            if code >= 400:
                logger.debug(f"HTTP {code} {args[0]}")
        except (ValueError, IndexError):
            pass


class APIServer:
    """Servidor HTTP rodando em thread separada (compatível com Windows e Linux)."""
    
    def __init__(self, port: int, data_dir: Path):
        self.port = port
        self.data_dir = data_dir
        self._server = None
        self._thread = None
        self._running = False
    
    def start(self):
        """Inicia o servidor em background."""
        handler = type("Handler", (TrafficHTTPHandler,), {
            "data_dir": self.data_dir,
        })
        
        self._server = HTTPServer(("0.0.0.0", self.port), handler)
        self._server.timeout = 1.0
        self._running = True
        
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        
        logger.info(f"API rodando em http://0.0.0.0:{self.port}")
        logger.info(f"  Metrics: http://localhost:{self.port}/metrics")
        logger.info(f"  Photos:  http://localhost:{self.port}/photos/{{name}}")
        logger.info(f"  Clips:   http://localhost:{self.port}/clips/{{name}}")
        logger.info(f"  Status:  http://localhost:{self.port}/status")
    
    def _run(self):
        """Loop do servidor — compatível com Windows e Linux."""
        import socket
        self._server.socket.settimeout(1.0)
        
        while self._running:
            try:
                self._server.handle_request()
            except Exception:
                if self._running:
                    continue
                break
    
    def stop(self):
        self._running = False
        if self._server:
            try:
                self._server.server_close()
            except Exception:
                pass
        logger.info("API parada")
