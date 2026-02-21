from __future__ import annotations

import ast
import asyncio
from dataclasses import dataclass
from typing import Any

from app.agent.models import ToolResult
from app.agent.tools.base import ToolRuntimeContext
from app.profiles.models import AgentProfile


_ALLOWED_BIN_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
_ALLOWED_UNARY_OPS = (ast.UAdd, ast.USub)
_ALLOWED_CALLS = {"abs": abs, "round": round, "min": min, "max": max}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    ast.Load,
    ast.Name,
    ast.Constant,
    ast.Tuple,
    ast.List,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


class _SafeAstEvaluator:
    def __init__(self, *, variables: dict[str, Any], max_operations: int) -> None:
        self._variables = variables
        self._max_operations = max(10, int(max_operations))
        self._operations = 0

    def eval(self, expression: str) -> Any:
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                raise ValueError(f"forbidden_node:{type(node).__name__}")
            if isinstance(node, ast.BinOp) and not isinstance(node.op, _ALLOWED_BIN_OPS):
                raise ValueError(f"forbidden_binop:{type(node.op).__name__}")
            if isinstance(node, ast.UnaryOp) and not isinstance(node.op, _ALLOWED_UNARY_OPS):
                raise ValueError(f"forbidden_unaryop:{type(node.op).__name__}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name):
                    raise ValueError("forbidden_call_target")
                if node.func.id not in _ALLOWED_CALLS:
                    raise ValueError(f"forbidden_call:{node.func.id}")
        return self._eval_node(tree.body)

    def _tick(self) -> None:
        self._operations += 1
        if self._operations > self._max_operations:
            raise ValueError("operation_limit_exceeded")

    def _eval_node(self, node: ast.AST) -> Any:
        self._tick()
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("non_numeric_constant")
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in self._variables:
                raise ValueError(f"unknown_variable:{node.id}")
            value = self._variables[node.id]
            if not isinstance(value, (int, float)):
                raise ValueError(f"non_numeric_variable:{node.id}")
            return value
        if isinstance(node, ast.Tuple):
            return tuple(self._eval_node(item) for item in node.elts)
        if isinstance(node, ast.List):
            return [self._eval_node(item) for item in node.elts]
        if isinstance(node, ast.UnaryOp):
            value = self._eval_node(node.operand)
            if isinstance(node.op, ast.UAdd):
                return +value
            return -value
        if isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Mod):
                return left % right
            return left**right
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name)
            fn = _ALLOWED_CALLS[node.func.id]
            args = [self._eval_node(item) for item in node.args]
            return fn(*args)
        raise ValueError(f"unsupported_node:{type(node).__name__}")


def _tool_policy(profile: AgentProfile | None) -> tuple[int, int, int]:
    if profile is None:
        return (256, 250, 1000)
    policy = profile.capabilities.tool_policies.get("python_calculator")
    if policy is None:
        return (256, 250, 1000)
    return (
        int(policy.max_expression_chars),
        int(policy.timeout_ms),
        int(policy.max_operations),
    )


@dataclass(frozen=True)
class PythonCalculatorTool:
    name: str = "python_calculator"

    async def run(
        self,
        payload: dict[str, object],
        *,
        state: dict[str, object],
        context: ToolRuntimeContext,
    ) -> ToolResult:
        del context
        profile = state.get("agent_profile")
        profile_obj = profile if isinstance(profile, AgentProfile) else None
        max_chars, timeout_ms, max_operations = _tool_policy(profile_obj)

        expression = str(
            payload.get("expression") or payload.get("code_snippet") or ""
        ).strip()
        if not expression:
            return ToolResult(tool=self.name, ok=False, error="missing_expression")
        if len(expression) > max_chars:
            return ToolResult(tool=self.name, ok=False, error="expression_too_long")
        variables_raw = payload.get("variables")
        variables: dict[str, Any] = (
            dict(variables_raw)
            if isinstance(variables_raw, dict)
            else {}
        )

        evaluator = _SafeAstEvaluator(
            variables=variables,
            max_operations=max_operations,
        )
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(evaluator.eval, expression),
                timeout=max(0.02, timeout_ms / 1000.0),
            )
        except TimeoutError:
            return ToolResult(tool=self.name, ok=False, error="calculation_timeout")
        except Exception as exc:
            return ToolResult(tool=self.name, ok=False, error=str(exc))

        if not isinstance(result, (int, float)):
            return ToolResult(tool=self.name, ok=False, error="non_numeric_result")
        return ToolResult(
            tool=self.name,
            ok=True,
            output={
                "expression": expression,
                "result": float(result),
            },
        )
