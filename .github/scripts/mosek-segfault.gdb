set pagination off
set print thread-events off

python
import gdb

def save_segfault_diagnostics(event):
    if not isinstance(event, gdb.SignalEvent) or event.stop_signal != "SIGSEGV":
        return
    gdb.execute("thread apply all bt full")
    gdb.execute("generate-core-file /tmp/pyoframe-mosek-segfault.core")

gdb.events.stop.connect(save_segfault_diagnostics)
end

run
