"""
Flask микросервер
"""

import json
import logging
import time

from src import __version__
from src import model

from flask import Flask, jsonify, request, g

logger = logging.getLogger('credit_default.api')

def json_log(payload: dict[str, any]) -> None:
    logger.info(json.dumps(payload, ensure_ascii=False))

def create_app() -> Flask:
    app = Flask(__name__)

    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    @app.before_request
    def _start_timer() -> None:
        g._start = time.perf_counter()

    @app.after_request
    def _access_log(responce):
        elapsed = round((time.perf_counter() - g._start) * 1000, 2)
        entry = {
            "event": "http_request",
            "method": request.method,
            "path": request.path,
            "status": responce.status_code,
            "latency": elapsed,
        }

        if getattr(g, "prediction", None) is not None:
            entry["prediction"] = g.prediction
            entry["prob_def"] = g.prob_def
        
        json_log(entry)
        return responce
    
    @app.get("/health")
    def health():
        return jsonify(status="Ok", service="credit-api", version=__version__)
    
    @app.post("/predict")
    def predict():
        if not request.is_json:
            return jsonify(error="Expected content type - json"), 400
        body = request.get_json(silent=True)
        if (body is None) or (not isinstance(body, dict)):
            return jsonify(error="Body must be a json pbject"), 400
        
        try:
            model.load()
            y_pred, prob = model.predict(body)
        except FileNotFoundError as e:
            return jsonify(error=str(e)), 503
        except ValueError as e:
            return jsonify(error=str(e)), 400
        g.prediction = y_pred
        g.prob_def = prob
        return jsonify(
            prediction=y_pred,
            probability_default=prob,
            model_version=__version__
        )
    
    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)