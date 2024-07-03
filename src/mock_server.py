from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/block_ip', methods=['POST'])
def block_ip():
    data = request.json
    ip_address = data.get('ip_address')
    if ip_address:
        return jsonify({"status": "success", "message": f"IP address {ip_address} blocked"}), 200
    else:
        return jsonify({"status": "error", "message": "IP address not provided"}), 400

@app.route('/api/allow_ip', methods=['POST'])
def allow_ip():
    data = request.json
    ip_address = data.get('ip_address')
    if ip_address:
        return jsonify({"status": "success", "message": f"IP address {ip_address} allowed"}), 200
    else:
        return jsonify({"status": "error", "message": "IP address not provided"}), 400

@app.route('/api/rate_limit', methods=['POST'])
def rate_limit():
    data = request.json
    ip_address = data.get('ip_address')
    rate_limit = data.get('rate_limit')
    if ip_address and rate_limit:
        return jsonify({"status": "success", "message": f"Rate limit {rate_limit} applied to IP address {ip_address}"}), 200
    else:
        return jsonify({"status": "error", "message": "IP address or rate limit not provided"}), 400

@app.route('/api/isolate_system', methods=['POST'])
def isolate_system():
    data = request.json
    system_id = data.get('system_id')
    if system_id:
        return jsonify({"status": "success", "message": f"System {system_id} isolated"}), 200
    else:
        return jsonify({"status": "error", "message": "System ID not provided"}), 400

@app.route('/api/send_alert', methods=['POST'])
def send_alert():
    data = request.json
    message = data.get('message')
    if message:
        return jsonify({"status": "success", "message": f"Alert sent: {message}"}), 200
    else:
        return jsonify({"status": "error", "message": "Message not provided"}), 400

@app.route('/api/trigger_malware_scan', methods=['POST'])
def trigger_malware_scan():
    data = request.json
    system_id = data.get('system_id')
    if system_id:
        return jsonify({"status": "success", "message": f"Malware scan triggered on system {system_id}"}), 200
    else:
        return jsonify({"status": "error", "message": "System ID not provided"}), 400

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    data = request.json
    settings = data.get('settings')
    if settings:
        return jsonify({"status": "success", "message": "Firewall settings updated"}), 200
    else:
        return jsonify({"status": "error", "message": "Settings not provided"}), 400

@app.route('/api/update_packages', methods=['POST'])
def update_packages():
    data = request.json
    system_id = data.get('system_id')
    if system_id:
        return jsonify({"status": "success", "message": f"Software packages updated on system {system_id}"}), 200
    else:
        return jsonify({"status": "error", "message": "System ID not provided"}), 400

@app.route('/api/search_logs', methods=['POST'])
def search_logs():
    data = request.json
    query = data.get('query')
    if query:
        return jsonify({"status": "success", "message": f"Log search performed with query: {query}"}), 200
    else:
        return jsonify({"status": "error", "message": "Query not provided"}), 400

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    return jsonify({"status": "success", "message": "Security report generated"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
