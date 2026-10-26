"""Minimal Windows pseudo-console (ConPTY) runner, stdlib + ctypes only.

Used to open an interactive CLI (agy) for a few seconds WITHOUT sending a prompt, so that its
official status-line hook fires with real quota data. Never used to drive a model.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from ctypes import wintypes

_EXTENDED_STARTUPINFO_PRESENT = 0x00080000
_CREATE_UNICODE_ENVIRONMENT = 0x00000400
_PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x00020016
_STARTF_USESTDHANDLES = 0x00000100
_S_OK = 0


class ConPtyError(OSError):
    pass


# ctypes reads `_fields_` as the structure layout (RUF012's ClassVar advice does not apply).
class _COORD(ctypes.Structure):
    _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]  # noqa: RUF012


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR), ("lpDesktop", wintypes.LPWSTR),  # noqa: RUF012
                ("lpTitle", wintypes.LPWSTR), ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
                ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD), ("dwXCountChars", wintypes.DWORD),
                ("dwYCountChars", wintypes.DWORD), ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
                ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD), ("lpReserved2", ctypes.c_void_p),
                ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE), ("hStdError", wintypes.HANDLE)]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]  # noqa: RUF012


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),  # noqa: RUF012
                ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD)]


def _k32() -> ctypes.WinDLL:
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreatePipe.argtypes = (ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(wintypes.HANDLE), ctypes.c_void_p,
                             wintypes.DWORD)
    k.CreatePseudoConsole.argtypes = (_COORD, wintypes.HANDLE, wintypes.HANDLE, wintypes.DWORD,
                                      ctypes.POINTER(ctypes.c_void_p))
    k.CreatePseudoConsole.restype = ctypes.c_long  # HRESULT
    k.ClosePseudoConsole.argtypes = (ctypes.c_void_p,)
    k.InitializeProcThreadAttributeList.argtypes = (ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD,
                                                    ctypes.POINTER(ctypes.c_size_t))
    k.UpdateProcThreadAttribute.argtypes = (ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t, ctypes.c_void_p,
                                            ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p)
    k.DeleteProcThreadAttributeList.argtypes = (ctypes.c_void_p,)
    k.CreateProcessW.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p, wintypes.BOOL,
                                 wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(_STARTUPINFOEXW),
                                 ctypes.POINTER(_PROCESS_INFORMATION))
    k.ReadFile.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                           ctypes.c_void_p)
    k.WriteFile.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD),
                            ctypes.c_void_p)
    k.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    k.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    k.CloseHandle.argtypes = (wintypes.HANDLE,)
    return k


def run_in_conpty(
    argv: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_s: float,
    done: Callable[[str], bool] = lambda _out: False, keystrokes: Sequence[tuple[str, str]] = (),
    cols: int = 160, rows: int = 50,
) -> str:
    """Run `argv` attached to a pseudo-console; return its (ANSI) output.

    `keystrokes` are (wait_for, text) pairs: `text` is typed only once `wait_for` appears in the
    ANSI-stripped output produced after the previous keystroke, so an unexpected dialog (trust
    prompt, update notice) never receives input meant for the main prompt. Stops when
    `done(output_so_far)` is true or `timeout_s` elapses; the process tree is always terminated.
    """
    k = _k32()
    in_read, in_write, out_read, out_write = (wintypes.HANDLE() for _ in range(4))
    if not k.CreatePipe(ctypes.byref(in_read), ctypes.byref(in_write), None, 0) or \
            not k.CreatePipe(ctypes.byref(out_read), ctypes.byref(out_write), None, 0):
        raise ConPtyError(ctypes.get_last_error(), "CreatePipe failed")
    hpc = ctypes.c_void_p()
    hr = k.CreatePseudoConsole(_COORD(cols, rows), in_read, out_write, 0, ctypes.byref(hpc))
    k.CloseHandle(in_read)
    k.CloseHandle(out_write)  # the pseudo-console owns its ends now
    if hr != _S_OK:
        raise ConPtyError(hr, "CreatePseudoConsole failed")

    size = ctypes.c_size_t()
    k.InitializeProcThreadAttributeList(None, 1, 0, ctypes.byref(size))
    attrs = ctypes.create_string_buffer(size.value)
    si = _STARTUPINFOEXW()
    si.StartupInfo.cb = ctypes.sizeof(_STARTUPINFOEXW)
    # Null std handles + STARTF_USESTDHANDLES: otherwise a child of a process whose stdio is redirected
    # (pipes, as under uv / an agent) inherits those handles and bypasses the pseudo-console entirely.
    si.StartupInfo.dwFlags = _STARTF_USESTDHANDLES
    si.lpAttributeList = ctypes.cast(attrs, ctypes.c_void_p)
    pi = _PROCESS_INFORMATION()
    chunks: list[str] = []
    try:
        if not k.InitializeProcThreadAttributeList(si.lpAttributeList, 1, 0, ctypes.byref(size)) or \
                not k.UpdateProcThreadAttribute(si.lpAttributeList, 0, _PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE,
                                                hpc, ctypes.sizeof(ctypes.c_void_p), None, None):
            raise ConPtyError(ctypes.get_last_error(), "pseudo-console attribute failed")
        block = "".join(f"{k_}={v}\0" for k_, v in sorted(env.items())) + "\0"
        cmdline = ctypes.create_unicode_buffer(subprocess.list2cmdline(list(argv)))
        if not k.CreateProcessW(None, cmdline, None, None, False,
                                _EXTENDED_STARTUPINFO_PRESENT | _CREATE_UNICODE_ENVIRONMENT,
                                ctypes.create_unicode_buffer(block), cwd, ctypes.byref(si), ctypes.byref(pi)):
            raise ConPtyError(ctypes.get_last_error(), f"CreateProcessW failed for {argv[0]}")

        def reader() -> None:  # must drain continuously or the child blocks on a full pipe
            buf = ctypes.create_string_buffer(65536)
            n = wintypes.DWORD()
            while k.ReadFile(out_read, buf, len(buf), ctypes.byref(n), None) and n.value:
                chunks.append(buf.raw[: n.value].decode("utf-8", "replace"))

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        start = time.monotonic()
        pending = list(keystrokes)
        mark = 0  # chunks index after the last keystroke
        while time.monotonic() - start < timeout_s:
            if k.WaitForSingleObject(pi.hProcess, 100) == 0:  # exited
                break
            if pending and pending[0][0] in strip_ansi("".join(chunks[mark:])):
                time.sleep(0.3)  # let the TUI finish drawing before typing
                data = pending.pop(0)[1].encode("utf-8")
                written = wintypes.DWORD()
                k.WriteFile(in_write, data, len(data), ctypes.byref(written), None)
                mark = len(chunks)
            if done("".join(chunks)):
                break
    finally:
        if pi.hProcess:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pi.dwProcessId)], capture_output=True, check=False)
            k.CloseHandle(pi.hThread)
            k.CloseHandle(pi.hProcess)
        k.ClosePseudoConsole(hpc)  # closes the output pipe's write side -> reader ends
        k.DeleteProcThreadAttributeList(si.lpAttributeList)
        k.CloseHandle(in_write)
        if "thread" in locals():
            thread.join(timeout=5)
        k.CloseHandle(out_read)
    return "".join(chunks)


def strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[=>()][0-9A-Za-z]?", "", text)


if os.name != "nt":  # pragma: no cover - the orchestrator targets Windows 11
    def run_in_conpty(*_a: object, **_k: object) -> str:  # type: ignore[no-redef]
        raise ConPtyError("ConPTY is Windows-only")
