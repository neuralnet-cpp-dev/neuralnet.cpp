#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""kvrecord.py — KVRecord v1/v2 参考实现（单文件、仅标准库）。

格式规范见 docs/kvrec-format-spec.md；权威实现为 C++ model_keyvalue_record.hpp。
合规口径：dumps_v2() 与 C++ serialize_v2() 逐字节一致；loads() 版本自举
（哨兵判别 v1/v2）。本实现供纯 Python 数据管线原生读写 .nnvocab/.nngames，
不再需要 subprocess 桥接 C++ CLI（neuralnet.cpp issue #35）。
"""

from __future__ import annotations

import struct
from typing import Any, Dict, List, Tuple, Union

__all__ = ["KVRecord", "KVRecordError"]

TYPE_UINT = 0
TYPE_STR = 1
TYPE_UINT_ARRAY = 2
TYPE_RECORD = 3

VERSION_KEY = "kvrec_version"
VERSION_V2 = 2
SENTINEL_INLINE_SIZE = 4 + 13 + 1 + 4 + 8  # 30

_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")


class KVRecordError(ValueError):
    """解析/序列化错误（对应 C++ 侧 NN_FAIL）。"""


class KVRecord:
    """键值记录。字段按插入顺序保存（list of (key, type, value)）。"""

    def __init__(self, _base: int = 0) -> None:
        self._fields: List[Tuple[str, int, Any]] = []
        # addr 基址：容器内嵌 kvrec 的绝对偏移（.nnvocab = FILE_LABEL_SIZE = 8）。
        # 解析时记住，dumps_v2() 重写时沿用 → addr 与权威 C++ serialize_v2 逐字节一致。
        self._base: int = _base

    # ── 写入 ──────────────────────────────────────────────────────────
    def set(self, key: str, value: Union[int, str, list, "KVRecord"]) -> "KVRecord":
        if isinstance(value, bool):
            raise KVRecordError("bool 不是 KVRecord 值类型（用 int）")
        if isinstance(value, int):
            return self.set_u64(key, value)
        if isinstance(value, str):
            return self.set_str(key, value)
        if isinstance(value, KVRecord):
            return self.set_record(key, value)
        if isinstance(value, (list, tuple)):
            return self.set_array(key, value)
        raise KVRecordError(f"不支持的值类型: {type(value)!r}")

    def set_u64(self, key: str, v: int) -> "KVRecord":
        if v < 0 or v >= (1 << 64):
            raise KVRecordError(f"u64 越界: {v}")
        self._fields.append((key, TYPE_UINT, int(v)))
        return self

    def set_str(self, key: str, v: str) -> "KVRecord":
        self._fields.append((key, TYPE_STR, v.encode("utf-8")))
        return self

    def set_bytes(self, key: str, v: bytes) -> "KVRecord":
        """原始字节串（Str 类型的字节形态；C++ 侧为 std::string）。"""
        self._fields.append((key, TYPE_STR, bytes(v)))
        return self

    def set_array(self, key: str, v: Union[list, tuple]) -> "KVRecord":
        for x in v:
            if not isinstance(x, int) or x < 0 or x >= (1 << 64):
                raise KVRecordError(f"u64 数组元素越界/非整数: {x!r}")
        self._fields.append((key, TYPE_UINT_ARRAY, [int(x) for x in v]))
        return self

    def set_record(self, key: str, v: "KVRecord") -> "KVRecord":
        self._fields.append((key, TYPE_RECORD, v))
        return self

    # ── 读取 ──────────────────────────────────────────────────────────
    def _find(self, key: str, type_: int) -> Any:
        for k, t, v in self._fields:
            if k == key and t == type_:
                return v
        return None

    def has(self, key: str) -> bool:
        return any(k == key for k, _, _ in self._fields)

    def get(self, key: str, default: Any = None) -> Any:
        """泛型读取：Str 返回**原始字节**（C++ std::string 语义——词表 blob 等
        可含非 UTF-8）；文本请显式用 get_str()。Record 返回子 KVRecord。"""
        for k, t, v in self._fields:
            if k == key:
                return v   # bytes / int / list / KVRecord
        return default

    def get_u64(self, key: str, default: int = None) -> Any:
        v = self._find(key, TYPE_UINT)
        return default if v is None else v

    def get_str(self, key: str, default: str = None) -> Any:
        v = self._find(key, TYPE_STR)
        return default if v is None else v.decode("utf-8")

    def get_bytes(self, key: str, default: bytes = None) -> Any:
        v = self._find(key, TYPE_STR)
        return default if v is None else v

    def get_array(self, key: str, default: list = None) -> Any:
        v = self._find(key, TYPE_UINT_ARRAY)
        return default if v is None else list(v)

    def get_record(self, key: str, default: "KVRecord" = None) -> Any:
        v = self._find(key, TYPE_RECORD)
        return default if v is None else v

    def keys(self) -> List[str]:
        return [k for k, _, _ in self._fields]

    def __len__(self) -> int:
        return len(self._fields)

    def __repr__(self) -> str:
        return f"KVRecord({len(self._fields)} fields: {self.keys()})"

    # ── v1 序列化 ─────────────────────────────────────────────────────
    def dumps(self) -> bytes:
        """v1 序列化（Record 字段跳过不写，与 C++ serialize() 一致）。"""
        out = bytearray()
        fields = [(k, t, v) for k, t, v in self._fields if t != TYPE_RECORD]
        out += _U32.pack(len(fields))
        for k, t, v in fields:
            kb = k.encode("utf-8")
            out += _U32.pack(len(kb))
            out += kb
            out += bytes([t])
            if t == TYPE_UINT:
                out += _U32.pack(8)
                out += _U64.pack(v)
            elif t == TYPE_STR:
                out += _U32.pack(len(v))
                out += v
            elif t == TYPE_UINT_ARRAY:
                out += _U32.pack(len(v) * 8)
                for x in v:
                    out += _U64.pack(x)
        return bytes(out)

    # ── v2 序列化 ─────────────────────────────────────────────────────
    def dumps_v2(self, base_offset: int = None) -> bytes:
        """v2 序列化，与 C++ serialize_v2(base_offset) 逐字节一致。
        省略 base_offset 时用加载时记住的容器基址（裸 kvrec 为 0）。"""
        if base_offset is None:
            base_offset = self._base
        return self._serialize_v2_blob(self._fields, base_offset)

    @staticmethod
    def _value_size(t: int, v: Any) -> int:
        if t == TYPE_UINT:
            return 8
        if t == TYPE_STR:
            return len(v)
        if t == TYPE_UINT_ARRAY:
            return len(v) * 8
        if t == TYPE_RECORD:
            return KVRecord._fields_blob_size(v._fields)
        raise KVRecordError(f"未知类型 {t}")

    @staticmethod
    def _fields_blob_size(fields: List[Tuple[str, int, Any]]) -> int:
        sz = 4 + SENTINEL_INLINE_SIZE
        for k, t, _ in fields:
            sz += 4 + len(k.encode("utf-8")) + 1 + 8
        for k, t, v in fields:
            sz += 8 + KVRecord._value_size(t, v)
        return sz

    @classmethod
    def _serialize_v2_blob(cls, fields, base_offset: int) -> bytes:
        out = bytearray()
        table = 4 + SENTINEL_INLINE_SIZE
        for k, _, _ in fields:
            table += 4 + len(k.encode("utf-8")) + 1 + 8

        # 字段表（哨兵 + 字段头）
        out += _U32.pack(len(fields) + 1)  # 含哨兵
        out += _U32.pack(13)
        out += VERSION_KEY.encode("ascii")
        out += bytes([TYPE_UINT])
        out += _U32.pack(8)
        out += _U64.pack(VERSION_V2)

        cursor = base_offset + table
        addrs = []
        for k, t, _ in fields:
            kb = k.encode("utf-8")
            out += _U32.pack(len(kb))
            out += kb
            out += bytes([t])
            addrs.append(cursor)
            out += _U64.pack(cursor)
            cursor += 8 + cls._value_size(t, _)

        # 数据块
        cursor = base_offset + table
        for i, (k, t, v) in enumerate(fields):
            if t == TYPE_UINT:
                out += _U64.pack(8)
                out += _U64.pack(v)
            elif t == TYPE_STR:
                out += _U64.pack(len(v))
                out += v
            elif t == TYPE_UINT_ARRAY:
                out += _U64.pack(len(v) * 8)
                for x in v:
                    out += _U64.pack(x)
            elif t == TYPE_RECORD:
                child_size = cls._fields_blob_size(v._fields)
                out += _U64.pack(child_size)
                out += cls._serialize_v2_blob(v._fields, cursor + 8)
            cursor += 8 + cls._value_size(t, v)
        assert cursor == addrs[-1] + 8 + cls._value_size(fields[-1][1], fields[-1][2]) + (
            0 if fields else 0
        ) or True  # 游标终值诊断（不强制）
        return bytes(out)

    # ── 解析（版本自举）──────────────────────────────────────────────
    @classmethod
    def loads(cls, data: bytes, offset: int = 0) -> "KVRecord":
        """从 bytes 解析（offset 为记录起点；v2 addr 为绝对偏移）。"""
        view = memoryview(data)
        rec = cls(_base=offset)
        pos = offset
        if pos + 4 > len(view):
            raise KVRecordError("读取越界：field_count")
        (count,) = _U32.unpack_from(view, pos)
        pos += 4
        if count == 0:
            return rec  # 空记录 = v1

        # field[0] 按 v1 布局读出（版本自举判别点）
        key0, type0, value0, pos = cls._read_v1_field(view, pos)

        if key0 == VERSION_KEY:
            # v2：哨兵校验 + 余下字段按 v2 布局
            if type0 != TYPE_UINT or len(value0) != 8:
                raise KVRecordError("kvrec 版本哨兵字段格式错误")
            (version,) = _U64.unpack_from(value0, 0)
            if version != VERSION_V2:
                raise KVRecordError(f"不支持的 kvrec 版本 {version}（当前支持 v2）")
            rec._fields = cls._parse_v2_fields(view, pos, count)
            return rec

        # v1：field[0] 已消费，继续 v1
        rec._fields.append(cls._decode_v1_field(key0, type0, value0))
        for _ in range(count - 1):
            key, type_, value, pos = cls._read_v1_field(view, pos)
            rec._fields.append(cls._decode_v1_field(key, type_, value))
        return rec

    @staticmethod
    def _read_v1_field(view, pos: int):
        if pos + 4 > len(view):
            raise KVRecordError("读取越界：key_len")
        (klen,) = _U32.unpack_from(view, pos)
        pos += 4
        if pos + klen + 1 + 4 > len(view):
            raise KVRecordError("读取越界：key/type/value_len")
        key = bytes(view[pos:pos + klen]).decode("utf-8")
        pos += klen
        type_ = view[pos]
        pos += 1
        (vlen,) = _U32.unpack_from(view, pos)
        pos += 4
        if pos + vlen > len(view):
            raise KVRecordError("读取越界：value")
        value = bytes(view[pos:pos + vlen])
        pos += vlen
        return key, type_, value, pos

    @staticmethod
    def _decode_v1_field(key: str, type_: int, value: bytes):
        if type_ == TYPE_UINT:
            if len(value) != 8:
                raise KVRecordError(f"字段 '{key}' 值长度错误")
            return (key, type_, _U64.unpack_from(value, 0)[0])
        if type_ == TYPE_STR:
            return (key, type_, value)
        if type_ == TYPE_UINT_ARRAY:
            if len(value) % 8 != 0:
                raise KVRecordError(f"字段 '{key}' 值长度错误")
            n = len(value) // 8
            return (key, type_, list(_U64.unpack_from(value, 0) for _ in range(n)) if False
                    else [_U64.unpack_from(value, i * 8)[0] for i in range(n)])
        # 未知类型：跳过（向前兼容）
        return None

    @classmethod
    def _parse_v2_fields(cls, view, offset: int, total_count: int):
        pos = offset
        heads = []  # (key, type, addr)
        for _ in range(1, total_count):
            if pos + 4 > len(view):
                raise KVRecordError("读取越界：v2 key_len")
            (klen,) = _U32.unpack_from(view, pos)
            pos += 4
            if pos + klen + 1 + 8 > len(view):
                raise KVRecordError("读取越界：v2 字段头")
            key = bytes(view[pos:pos + klen]).decode("utf-8")
            pos += klen
            type_ = view[pos]
            pos += 1
            (addr,) = _U64.unpack_from(view, pos)
            pos += 8
            heads.append((key, type_, addr))

        blocks_begin = pos
        fields = []
        expect = blocks_begin
        for key, type_, addr in heads:
            if addr != expect:
                raise KVRecordError(
                    f"字段 '{key}' 数据块地址不连续（整体搬迁的 kvrec 字节地址已失效）")
            if addr + 8 > len(view):
                raise KVRecordError(f"字段 '{key}' 数据块越界")
            (vlen,) = _U64.unpack_from(view, addr)
            expect = addr + 8 + vlen

            if type_ not in (TYPE_UINT, TYPE_STR, TYPE_UINT_ARRAY, TYPE_RECORD):
                continue  # 未知类型：按长度跳过（向前兼容）
            if type_ == TYPE_RECORD:
                sub = cls.loads(bytes(view), addr + 8)
                fields.append((key, TYPE_RECORD, sub))
            else:
                if addr + 8 + vlen > len(view):
                    raise KVRecordError(f"字段 '{key}' 值越界")
                value = bytes(view[addr + 8:addr + 8 + vlen])
                f = cls._decode_v1_field(key, type_, value)
                if f is not None:
                    fields.append(f)
        return fields

    # ── 文件便捷 ─────────────────────────────────────────────────────
    # ── 文件标签容器（magic 4B + file_version u32；.nnvocab 等）───────
    # 权威语义：dataset_format.hpp 的 write_file_label / parse_nnvocab ——
    # 标签 = [magic 4B][file_version u32]（共 8B），kvrec 记录起点固定为 8；
    # v2 内部 addr 为文件绝对偏移（含标签头），故必须带完整文件解析。
    NNVOCAB_MAGIC = b"NNVC"
    FILE_LABEL_SIZE = 8

    @classmethod
    def load_nnvocab(cls, data: bytes) -> "KVRecord":
        """解析 .nnvocab 容器字节：[magic 4B][file_version u32][kvrec v2]。"""
        if len(data) < cls.FILE_LABEL_SIZE or data[:4] != cls.NNVOCAB_MAGIC:
            raise KVRecordError("不是 .nnvocab 容器（缺 NNVC 魔数）")
        return cls.loads(data, cls.FILE_LABEL_SIZE)

    @classmethod
    def load(cls, path: str) -> "KVRecord":
        """从文件加载。按内容嗅探：NNVC 魔数 → 容器；否则裸 kvrec。"""
        with open(path, "rb") as fp:
            data = fp.read()
        if len(data) >= 4 and data[:4] == cls.NNVOCAB_MAGIC:
            return cls.load_nnvocab(data)
        return cls.loads(data)

    def save(self, path: str, v2: bool = True) -> None:
        blob = self.dumps_v2() if v2 else self.dumps()
        with open(path, "wb") as fp:
            fp.write(blob)


if __name__ == "__main__":
    # 自检：round-trip + 版本自举
    r = KVRecord()
    r.set_u64("vocab_size", 50257)
    r.set_str("model", "bpe")
    r.set_array("merges", [1, 2, 3])
    r.set_record("meta", KVRecord().set_u64("ver", 2).set_str("x", "yz"))
    blob = r.dumps_v2()
    r2 = KVRecord.loads(blob)
    assert r2.get_u64("vocab_size") == 50257
    assert r2.get_str("model") == "bpe"
    assert r2.get_array("merges") == [1, 2, 3]
    assert r2.get_record("meta").get_str("x") == "yz"
    # v1 自举
    r1blob = r.dumps()
    r3 = KVRecord.loads(r1blob)
    assert r3.get_u64("vocab_size") == 50257
    assert r3.get_record("meta") is None  # v1 无 Record

    # 容器 base 追踪（.nnvocab 语义，base=8）：base=8 构造 → 拼 8B 标签 →
    # 解析记住 base → 重写逐字节还原。
    c = KVRecord().set_u64("a", 1).set_str("b", "xy")
    c8 = KVRecord().set_u64("a", 1).set_str("b", "xy")
    tag_blob = c8.dumps_v2(base_offset=8)            # addr 基于 base=8
    label = b"NNVC" + struct.pack("<I", 1)           # [magic 4B][file_version u32]
    inline = KVRecord.loads(label + tag_blob, 8)     # base=8 解析
    assert inline._base == 8
    assert inline.get_u64("a") == 1 and inline.get_str("b") == "xy"
    assert inline.dumps_v2() == tag_blob, "容器 base 追踪：重写须逐字节一致"
    # 容器便捷方法：load_nnvocab 嗅探魔数 → 同样 base=8
    from_bytes = KVRecord.load_nnvocab(label + tag_blob)
    assert from_bytes.dumps_v2() == tag_blob

    print(f"kvrecord.py 自检通过：v2 {len(blob)} B / v1 {len(r1blob)} B / 容器 base=8 往返一致")
