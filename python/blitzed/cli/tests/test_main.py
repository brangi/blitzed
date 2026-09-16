from click.testing import CliRunner

from blitzed.cli.main import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_info_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["info"])
    assert result.exit_code == 0
    assert "Blitzed Information" in result.output


def test_missing_required_arg():
    runner = CliRunner()
    result = runner.invoke(cli, ["optimize"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output


def test_invalid_option():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "--invalid"])
    assert result.exit_code != 0
    assert "No such option" in result.output
