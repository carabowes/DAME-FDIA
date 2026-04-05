import numpy as np

from src.ml.mitigation_metrics import (
    extract_alarm_segments,
    compute_false_incident_rate,
    evaluate_episode_detection,
    summarise_episode_detection,
)

# Test 1: Alarm segmentation

def test_extract_alarm_segments_basic():
    alarm_mask = np.array([
        0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0
    ])

    segments = extract_alarm_segments(alarm_mask)

    expected = [(2, 5), (7, 9), (10, 11)]
    assert segments == expected

def test_extract_alarm_segments_empty():
    alarm_mask = np.zeros(50, dtype=int)
    segments = extract_alarm_segments(alarm_mask)

    assert segments == []

# Test 2: False incident rate

def test_false_incident_rate():
    """False incident rate = count of alarms NOT overlapping with attacks."""
    alarm_segments = [(10, 12), (40, 45), (300, 310)]  # 3 alarms
    attack_episodes = [(5, 8), (100, 150)]  # 2 attacks, don't overlap with any alarm
    T = 500

    result = compute_false_incident_rate(
        alarm_segments=alarm_segments,
        attack_episodes=attack_episodes,
        T=T
    )

    # All 3 alarms are false (occur outside attacks)
    assert result["false_incidents"] == 3
    assert result["false_incidents_per_500"] == 3 * (500 / 500)

def test_false_incident_rate_zero():
    """No false incidents when all alarms overlap with attacks."""
    alarm_segments = [(105, 110), (110, 115)]  # 2 alarms inside attack
    attack_episodes = [(100, 150)]
    T = 500

    result = compute_false_incident_rate(
        alarm_segments=alarm_segments,
        attack_episodes=attack_episodes,
        T=T
    )

    # No false incidents (both alarms overlap with attack)
    assert result["false_incidents"] == 0
    assert result["false_incidents_per_500"] == 0.0

# Test 3: Episode-level detection

def test_episode_detection_basic():
    """Detect attacks with appropriate TTFD (time-to-first-detection)."""
    attack_episodes = [(100, 150), (300, 350)]
    alarm_segments = [(90, 95), (110, 120), (360, 370)]

    results = evaluate_episode_detection(
        attack_episodes=attack_episodes,
        alarm_segments=alarm_segments
    )

    # Episode 1 (100-150): alarm at (110-120) overlaps, TTFD = 110 - 100 = 10
    assert results[0]["detected"] is True
    assert results[0]["ttfd"] == 10

    # Episode 2 (300-350): alarm at (360-370) doesn't overlap, not detected
    assert results[1]["detected"] is False
    assert results[1]["ttfd"] is None


def test_episode_detection_exact_start():
    """Alarm at exact episode start should have TTFD=0."""
    attack_episodes = [(50, 100)]
    alarm_segments = [(50, 60)]

    results = evaluate_episode_detection(
        attack_episodes=attack_episodes,
        alarm_segments=alarm_segments
    )

    assert results[0]["detected"] is True
    assert results[0]["ttfd"] == 0


def test_episode_detection_multiple_alarms():
    """Multiple alarms in same episode: choose first alarm for TTFD."""
    attack_episodes = [(200, 300)]
    alarm_segments = [(210, 215), (220, 225)]

    results = evaluate_episode_detection(
        attack_episodes=attack_episodes,
        alarm_segments=alarm_segments
    )

    # First alarm at 210, episode starts at 200, so TTFD = 10
    assert results[0]["detected"] is True
    assert results[0]["ttfd"] == 10

# Test 4: Episode summary statistics

def test_episode_summary():
    episode_results = [
        {"detected": True, "ttfd": 5},
        {"detected": False, "ttfd": None},
        {"detected": True, "ttfd": 20},
    ]

    summary = summarise_episode_detection(episode_results)

    assert summary["num_episodes"] == 3
    assert summary["num_detected"] == 2
    assert summary["detection_rate"] == 2 / 3
    assert summary["median_ttfd"] == 12.5

def test_episode_summary_no_detections():
    episode_results = [
        {"detected": False, "ttfd": None},
        {"detected": False, "ttfd": None},
    ]

    summary = summarise_episode_detection(episode_results)

    assert summary["num_detected"] == 0
    assert summary["median_ttfd"] is None
