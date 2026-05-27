def generate_explanation(state, attack_ratio, total_packets):
    explanation = ""

    if state == "ATTACK":
        explanation = f"""
🚨 HIGH RISK DETECTED

The system observed a high proportion of malicious traffic ({attack_ratio:.2f}).
This indicates sustained abnormal behavior consistent with IoT botnet activity.

Possible Causes:
- Distributed Denial of Service (DDoS)
- TCP/UDP Flooding
- Compromised IoT device sending repeated requests

Recommendation:
- Isolate affected device
- Monitor outgoing traffic
- Block suspicious IPs
"""

    elif state == "SUSPICIOUS":
        explanation = f"""
⚠️ MEDIUM RISK

Traffic shows irregular patterns ({attack_ratio:.2f} attack ratio).
This may indicate early-stage attack or scanning activity.

Possible Causes:
- Port scanning
- Initial botnet communication
- Misconfigured device

Recommendation:
- Monitor closely
- Log traffic for further analysis
"""

    else:
        explanation = f"""
✅ LOW RISK

Traffic appears normal with low anomaly ratio ({attack_ratio:.2f}).

System Behavior:
- No significant attack patterns detected
- Network operating within expected parameters

Recommendation:
- Continue monitoring
"""

    return explanation
