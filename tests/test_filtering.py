import unittest

import pandas as pd

from app.filtering import is_duplicate, match_metadata


class MatchMetadataTests(unittest.TestCase):
    def test_matches_when_video_file_column_contains_missing_values(self):
        df = pd.DataFrame(
            {
                "video_file": [None, "sample.mp4"],
                "location": ["placeholder", "공원"],
                "start_time": ["2024-01-01 00:00:00", "2024-01-01 01:00:00"],
            }
        )

        result = match_metadata("sample.mp4", df)

        self.assertTrue(result.matched)
        self.assertEqual(result.location, "공원")

    def test_returns_unmatched_when_video_file_column_is_missing(self):
        df = pd.DataFrame(
            {
                "location": ["공원"],
                "start_time": ["2024-01-01 00:00:00"],
            }
        )

        result = match_metadata("sample.mp4", df)

        self.assertFalse(result.matched)
        self.assertEqual(result.video_file, "sample.mp4")


if __name__ == "__main__":
    unittest.main()
