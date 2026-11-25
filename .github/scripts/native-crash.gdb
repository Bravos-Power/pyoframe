set pagination off
set print thread-events off

python
import gdb

def save_native_crash_diagnostics(event):
    if not isinstance(event, gdb.SignalEvent) or event.stop_signal not in ("SIGSEGV", "SIGILL"):
        return
    gdb.execute("thread apply all bt full")
    gdb.execute("generate-core-file /tmp/pyoframe-native-crash.core")

gdb.events.stop.connect(save_native_crash_diagnostics)
end

run
