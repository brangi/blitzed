import pytest
from click.testing import CliRunner
from blitzed.cli.main import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ['--help'])
    assert result.exit_code == 0
    assert 'Usage' in result.output

def test_valid_command():
    runner = CliRunner()
    result = runner.invoke(cli, ['command', '--option', 'value'])
    assert result.exit_code == 0
    assert 'Success!' in result.output

def test_missing_required_arg():
    runner = CliRunner()
    result = runner.invoke(cli, ['command'])
    assert result.exit_code != 0
    assert 'Missing argument' in result.output

def test_invalid_option():
    runner = CliRunner()
    result = runner.invoke(cli, ['command', '--invalid', 'value'])
    assert result.exit_code != 0
    assert 'no such option' in result.output