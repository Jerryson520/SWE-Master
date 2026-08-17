import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_source(relative_path: str) -> ast.Module:
    source_path = PROJECT_ROOT / relative_path
    return ast.parse(source_path.read_text(), filename=str(source_path))


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} was not found")


class StartupRegressionTests(unittest.TestCase):
    def test_prepull_worker_accepts_the_docker_ip_argument(self):
        tree = parse_source("src/r2egym/agenthub/run/edit.py")
        function = find_function(tree, "prepull_docker_image")

        argument_names = [argument.arg for argument in function.args.args]
        self.assertEqual(argument_names, ["docker_image", "ip"])
        self.assertEqual(len(function.args.defaults), 1)

    def test_make_test_spec_function_is_not_shadowed_by_a_local_variable(self):
        tree = parse_source("src/r2egym/agenthub/runtime/docker.py")
        initializer = find_function(tree, "__init__")

        local_assignments = {
            node.id
            for node in ast.walk(initializer)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        self.assertNotIn("make_test_spec", local_assignments)

    def test_local_docker_mode_does_not_force_a_tcp_daemon(self):
        tree = parse_source("src/r2egym/agenthub/runtime/docker.py")
        initializer = find_function(tree, "__init__")

        docker_host_assignments = [
            node
            for node in ast.walk(initializer)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and target.attr == "docker_host"
                for target in node.targets
            )
        ]
        self.assertEqual(len(docker_host_assignments), 1)
        self.assertIsInstance(docker_host_assignments[0].value, ast.IfExp)

    def test_swebench_image_validation_requires_custom_test_runner(self):
        tree = parse_source("src/r2egym/agenthub/run/edit.py")
        validator = find_function(tree, "is_valid_swebench_verified_image")
        validator_source = ast.unparse(validator)

        self.assertIn("SWEBENCH_VERIFIED_IMAGE_PREFIX", validator_source)
        self.assertIn("SWEBENCH_TEST_RUNNER", validator_source)
        self.assertIn("image_contains_file", validator_source)

    def test_failed_prepull_is_fatal(self):
        tree = parse_source("src/r2egym/agenthub/run/edit.py")
        function = find_function(tree, "prepull_docker_images")

        self.assertTrue(
            any(isinstance(node, ast.Raise) for node in ast.walk(function)),
            "prepull failures must stop the run instead of producing a false completion",
        )

    def test_missing_swebench_runner_aborts_environment_setup(self):
        tree = parse_source("src/r2egym/agenthub/runtime/docker.py")
        function = find_function(tree, "setup_env_swebench")
        function_source = ast.unparse(function)

        self.assertIn("test -f /run_tests.sh", function_source)
        self.assertGreaterEqual(
            sum(isinstance(node, ast.Raise) for node in ast.walk(function)), 2
        )


if __name__ == "__main__":
    unittest.main()
