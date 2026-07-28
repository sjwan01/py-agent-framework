# py_agent — Remaining Cleanup

所有之前指出的问题中，除了这一条，其余都已在结构清理阶段顺手修复。

---

## `_build_agent` 裸 `list` 类型参数

**文件**: `py_agent/runner/_agent.py`，第 224–225 行

```diff
-        pending: list | None = None,
-        streamers: list | None = None,
+        pending: list[Any] | None = None,
+        streamers: list[Any] | None = None,
```

文件中 `Any` 已 import（第 20 行）。

---

验证：`pytest tests/ -q && mypy --ignore-missing-imports py_agent/runner/_agent.py`

17 passed, no new mypy errors.
