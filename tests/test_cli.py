import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from parapc_fm.cli import parse_args


class CLITests(unittest.TestCase):
    def test_all_model_configs_and_cli_override(self):
        root = Path(__file__).resolve().parents[1]
        for model in ("flux", "sd3", "sd35", "sd15"):
            args = parse_args(model, ["--config", str(root / "configs" / f"{model}.yaml"),
                                      "--init_order", "2", "--seed", "42"])
            self.assertEqual(args.init_order, 2)
            self.assertEqual(args.seed, 42)
            self.assertEqual(args.window_size, 8)

    def test_yaml_orders_are_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "bad.yaml"
            config.write_text("init_order: 3\n")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args("flux", ["--config", str(config)])

    def test_sd15_rejects_flow_order(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args("sd15", ["--update_order", "1"])


if __name__ == "__main__":
    unittest.main()
