from flask import Flask, jsonify
import requests

app = Flask(__name__)

def get_ip_location():
    try:
        res = requests.get("https://ipinfo.io/json").json()
        if "loc" in res:
            lat, lon = res["loc"].split(",")
            return float(lat), float(lon)
        return None, None
    except:
        return None, None

@app.route("/get_gps")
def get_gps():
    lat, lon = get_ip_location()
    if lat is None or lon is None:
        return jsonify({"lat": None, "lon": None})
    return jsonify({"lat": lat, "lon": lon})

if __name__ == "__main__":
    app.run(port=5001)