from flask import Flask, request
from waitress import serve

from blueprints.geojson import geojson_api
import logging

logging.basicConfig(level=logging.WARNING)

# Flask configuratiosn
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

# Logging for incoming requests
@app.before_request
def log_request_info():
    app.logger.info(f'Request: {request.method} {request.url}')

# Logging levels
app.logger.info('Info level log')
app.logger.warning('Warning level log')
app.logger.error('Error level log')


# Route: /get_geojson
app.register_blueprint(geojson_api)

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000, threads=8)