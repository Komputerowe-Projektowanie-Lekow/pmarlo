"""Input/output utilities and configuration for PMARLO."""

# Whether to emit verbose messages from external trajectory plugins.
# Users may toggle this at runtime to inspect plugin-level diagnostics.
# By default, the VMD/MDTraj DCD plugin chatter is silenced to keep logs
# focused on PMARLO's own output.
verbose_plugin_logs: bool = False

__all__ = ["verbose_plugin_logs"]
