from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
import io
import base64
import os

app = Flask(__name__)
app.secret_key = 'tsa'
socketio = SocketIO(app)

class ThompsonSampling:
    def __init__(self, n_machines=3):
        self.n_machines = n_machines
        self.alpha = np.ones(n_machines)
        self.beta = np.ones(n_machines)
        self.selection_counts = np.zeros(n_machines, dtype=int)
        self.cumulative_rewards = 0
        self.true_rewards = np.random.uniform(0.01, 0.1, n_machines)

    def select_button(self):
        sampled_theta = np.random.beta(self.alpha, self.beta)
        return np.argmax(sampled_theta)

    def update(self, chosen_machine, reward):
        self.selection_counts[chosen_machine] += 1
        if reward == 1:
            self.alpha[chosen_machine] += 1
        else:
            self.beta[chosen_machine] += 1
        self.cumulative_rewards += reward

users = ['jerwin', 'renato', 'juan']

user_sessions = {}

def get_user_session(username):
    if username not in users:
        return None
    if username not in user_sessions:
        user_sessions[username] = {
            "ts": ThompsonSampling(),
            "selections": []
        }
    return user_sessions[username]

@app.route("/")
def index():
    return render_template("index.html", users=users)

@app.route("/suggest", methods=["GET"])
def suggest():
    username = request.args.get('user')
    user_session = get_user_session(username)
    if not user_session:
        return jsonify({"error": "Invalid user"}), 400
    ts = user_session['ts']
    suggested_button = ts.select_button()
    return jsonify({
        "suggested_button": int(suggested_button),
        "user": username
    })
    
    
machine_names = ["Email", "SMS", "Push Notification"]
def generate_visualization(ts, selections):
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs[0, 0].bar(range(ts.n_machines), ts.selection_counts, color='skyblue')
    axs[0, 0].set_title("Arm Selection Counts")
    axs[0, 0].set_xlabel("Arm")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_xticks(range(ts.n_machines))
    axs[0, 0].set_xticklabels(machine_names)
    
    x = np.linspace(0, 1, 100)
    for i in range(ts.n_machines):
        y = beta.pdf(x, ts.alpha[i], ts.beta[i])
        axs[0, 1].plot(x, y, label=machine_names[i])
    axs[0, 1].set_title("Beta Distributions")
    axs[0, 1].set_xlabel("Probability of Success")
    axs[0, 1].set_ylabel("Density")
    axs[0, 1].legend()
    

    if selections:
        axs[1, 0].plot(np.cumsum(selections), label="Cumulative Rewards")
    else:
        axs[1, 0].text(0.5, 0.5, 'No data yet', ha='center', va='center')
    axs[1, 0].set_title("Cumulative Rewards")
    axs[1, 0].set_xlabel("Trial")
    axs[1, 0].set_ylabel("Reward")
    axs[1, 0].legend()

    axs[1, 1].bar(range(ts.n_machines), ts.true_rewards, alpha=0.6, label='True Rewards')
    estimated = ts.alpha / (ts.alpha + ts.beta)
    axs[1, 1].bar(range(ts.n_machines), estimated, alpha=0.4, label='Estimated')
    axs[1, 1].set_title("True vs Estimated Rewards")
    axs[1, 1].set_xlabel("Arm")
    axs[1, 1].set_ylabel("Reward Probability")
    axs[1, 1].legend()
    axs[1, 1].set_xticks(range(ts.n_machines))
    axs[1, 1].set_xticklabels(machine_names)

    plt.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return base64.b64encode(img.getvalue()).decode('utf-8')

@app.route("/visualization", methods=["GET"])
def visualization():
    username = request.args.get('user')
    user_session = get_user_session(username)
    if not user_session:
        return jsonify({"error": "Invalid user"}), 400
    ts = user_session['ts']
    selections = user_session['selections']
    image_base64 = generate_visualization(ts, selections)
    return f"data:image/png;base64,{image_base64}"

@app.route("/update", methods=["POST"])
def update():
    data = request.json
    username = data.get("user")
    if not username or username not in users:
        return jsonify({"error": "Invalid user"}), 400
    user_session = get_user_session(username)
    ts = user_session['ts']
    chosen_button = int(data["button"])
    reward = int(data["reward"])
    ts.update(chosen_button, reward)
    user_session['selections'].append(reward)

    socketio.emit("update_visualization", {
        "user": username,
        "alpha": [float(a) for a in ts.alpha],
        "beta": [float(b) for b in ts.beta],
        "selection_counts": ts.selection_counts.tolist(),
        "visualization": generate_visualization(ts, user_session['selections']),
        "true_rewards": ts.true_rewards.tolist()
    })

    return jsonify({"status": "success", "user": username})

if __name__ == "__main__":
    socketio = SocketIO(app, async_mode='threading')
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), allow_unsafe_werkzeug=True)




