import pytest

from src.core.book_profiles.profile_goals import declared_profile_goal


@pytest.mark.parametrize(("config", "expected"), [
    ({"profile_goal": "  Faithful  ", "audiobook": {"enabled": True}}, "faithful_translation"),
    ({"profile_goal": "literary", "business_rules": {"goal": "academic"}}, "literary_polish"),
    ({"business_rules": {"goal": "modernizar"}, "goal": "translate"}, "modernization"),
    ({"goal": "technical"}, "academic_translation"),
    ({"transform_mode": "simplify"}, "explanatory_rewrite"),
    ({"mode": "literary_polish"}, "literary_polish"),
    ({"audiobook": {"enabled": True}, "transform_mode": "translate"}, "audiobook"),
    ({"audio_sanitization": {"generate_companion": True}}, "audiobook"),
    ({"audit_dimensions": "translation_audiobook_full"}, "audiobook"),
])
def test_declared_profile_goal_reads_saved_purpose(config, expected):
    assert declared_profile_goal(config).key == expected


@pytest.mark.parametrize("config", [
    {},
    {"profile_id": "audio_custom", "source_name": "Modern technical thesis.epub"},
    {"editorial_mode": "book_profile", "modernization_strength": "high"},
    {"business_rules": {"goal": "personal"}, "audiobook": {"enabled": True}},
    {"profile_goal": "custom", "business_rules": {"goal": "translate"}},
    {"audiobook": {"enabled": False}, "audit_dimensions": "editorial_full"},
    {"audio_sanitization": {"enabled": "false"}},
    {"business_rules": ["audio"], "mode": "custom"},
])
def test_declared_profile_goal_preserves_unclassified_or_custom_policy(config):
    assert declared_profile_goal(config) is None
