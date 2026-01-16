"""Real-time diagnostic for track state transitions."""
import sys
sys.path.insert(0, "src")

import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def analyze_event(event_str: str):
    """Analyze a track event and diagnose why it's classified as ENEMY."""

    print("=" * 80)
    print("TRACK STATE DIAGNOSTIC")
    print("=" * 80)

    try:
        event = json.loads(event_str)
    except:
        print("✗ Invalid JSON format")
        return

    print(f"\n📊 EVENT DETAILS:")
    print(f"   Track ID: {event.get('track_id')}")
    print(f"   State: {event.get('state')}")
    print(f"   Person ID: {event.get('person_id')}")
    print(f"   Confidence: {event.get('confidence')}")

    modalities = event.get('modalities_used', {})
    print(f"\n🎯 MODALITY SCORES:")
    for modality, score in modalities.items():
        print(f"   {modality}: {score:.4f}")

    # Analyze why ENEMY
    if event.get('state') == 'ENEMY':
        print(f"\n❌ CLASSIFIED AS ENEMY")
        print(f"\n🔍 DIAGNOSIS:")

        face_score = modalities.get('face', 0.0)

        print(f"\n1. Face Recognition Score: {face_score:.4f}")

        # Load config to check thresholds
        from watchbird.config import Config
        config = Config("config.yaml")

        t_accept = config.thresholds.get('t_accept', 0.65)
        t_margin = config.thresholds.get('t_margin', 0.10)
        t_timeout = config.thresholds.get('t_timeout', 5.0)
        consistency_count = config.fusion.get('consistency_count', 6)
        window_size = config.fusion.get('window_size', 10)

        print(f"\n2. Current Thresholds:")
        print(f"   t_accept (min score):        {t_accept}")
        print(f"   t_margin (min margin):       {t_margin}")
        print(f"   t_timeout (max time):        {t_timeout}s")
        print(f"   consistency_count (frames):  {consistency_count}")
        print(f"   window_size (buffer):        {window_size}")

        print(f"\n3. Failure Analysis:")

        reasons = []

        # Check if score meets threshold
        if face_score < t_accept:
            reasons.append(f"Score too low: {face_score:.4f} < {t_accept} (t_accept)")
            print(f"   ✗ Score: {face_score:.4f} is BELOW threshold {t_accept}")
        else:
            print(f"   ✓ Score: {face_score:.4f} is ABOVE threshold {t_accept}")

        # Likely timeout or consistency issue
        if face_score >= t_accept:
            reasons.append("Possible causes:")
            reasons.append(f"  - Timeout: Track exceeded {t_timeout}s without reaching {consistency_count} consistent frames")
            reasons.append(f"  - Margin too low: Score margin was < {t_margin}")
            reasons.append(f"  - Inconsistent detections: Fewer than {consistency_count} frames matched same person")

            print(f"   ⚠ Score is good, but classification still ENEMY")
            print(f"   → Most likely: TIMEOUT or CONSISTENCY failure")
            print(f"\n   Possible reasons:")
            print(f"   • Track was visible for less than {consistency_count} frames")
            print(f"   • Different person IDs detected across frames (inconsistent)")
            print(f"   • Margin between 1st and 2nd match was < {t_margin}")
            print(f"   • Timeout of {t_timeout}s was reached before consistency")

        print(f"\n💡 RECOMMENDED FIXES:")

        if face_score >= 0.50 and face_score < t_accept:
            print(f"\n   FIX 1: Lower t_accept threshold")
            print(f"   Current: {t_accept}")
            print(f"   Recommended: {max(0.45, face_score - 0.05):.2f}")
            print(f"   Edit config.yaml:")
            print(f"     thresholds:")
            print(f"       t_accept: {max(0.45, face_score - 0.05):.2f}")

        if face_score >= t_accept:
            print(f"\n   FIX 1: Reduce consistency_count")
            print(f"   Current: {consistency_count} frames")
            print(f"   Recommended: {max(3, consistency_count - 2)} frames")
            print(f"   Edit config.yaml:")
            print(f"     fusion:")
            print(f"       consistency_count: {max(3, consistency_count - 2)}")

            print(f"\n   FIX 2: Increase timeout")
            print(f"   Current: {t_timeout}s")
            print(f"   Recommended: {t_timeout + 3.0}s")
            print(f"   Edit config.yaml:")
            print(f"     thresholds:")
            print(f"       t_timeout: {t_timeout + 3.0}")

            print(f"\n   FIX 3: Lower margin requirement")
            print(f"   Current: {t_margin}")
            print(f"   Recommended: {max(0.05, t_margin - 0.03):.2f}")
            print(f"   Edit config.yaml:")
            print(f"     thresholds:")
            print(f"       t_margin: {max(0.05, t_margin - 0.03):.2f}")

        print(f"\n📝 QUICK FIX (Apply all):")
        print(f"   Edit config.yaml and change:")
        print(f"   ")
        print(f"   thresholds:")
        print(f"     t_accept: {max(0.45, min(t_accept, face_score - 0.05)):.2f}  # Current: {t_accept}")
        print(f"     t_margin: {max(0.05, t_margin - 0.03):.2f}  # Current: {t_margin}")
        print(f"     t_timeout: {t_timeout + 3.0}  # Current: {t_timeout}")
        print(f"   ")
        print(f"   fusion:")
        print(f"     consistency_count: {max(3, consistency_count - 2)}  # Current: {consistency_count}")

    elif event.get('state') == 'FRIENDLY':
        print(f"\n✓ CORRECTLY CLASSIFIED AS FRIENDLY")
        confidence = event.get('confidence', 0.0)
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Person: {event.get('person_id')}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        event_str = sys.argv[1]
    else:
        # Default to your provided event
        event_str = '{"ts": 1768601593.2730865, "track_id": 10, "state": "ENEMY", "person_id": null, "confidence": 0.0, "modalities_used": {"face": 0.6327513367067373}}'

    analyze_event(event_str)
