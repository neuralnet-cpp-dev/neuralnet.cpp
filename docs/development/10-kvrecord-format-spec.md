# KVRecord 二进制格式规范（v1 / v2）

> **读者**：调用者（读写 `.nnvocab` / `.nngames` 等容器的工具作者）——本分册定位见 [`README.md`](README.md)。
> **一句话**：KeyValueRecord 容器的字节级稳定契约；自 v2 起承诺同版本字节布局不变，演进只经"新增版本哨兵值"通道（§4）。
> **状态**：现行（格式契约 · 权威实现 [`model_keyvalue_record.hpp`](../../include/neuralnet.cpp/model_keyvalue_record.hpp)；Python 参考实现 [`scripts/kvrecord.py`](../../scripts/kvrecord.py)，单文件、仅标准库）。

**字节序：全程小端（little-endian）。** 所有整数均为无符号。

## 1. 值类型（两版本共用的类型标签）

| type | 名称 | 值语义 |
|---|---|---|
| `0` | UInt | 8 字节 uint64 |
| `1` | Str | 原始字符串字节（无编码假设，长度前缀） |
| `2` | UIntArray | `count × 8` 字节的 uint64 数组 |
| `3` | Record | **仅 v2**：完整嵌套 v2 kvrec 字节（含自身哨兵） |

**向前兼容**：解析器遇到未知 type 时按 `value_len` 长度**安全跳过**，不报错。

## 2. v1（内联布局）

```
[field_count u32]
field := [key_len u32][key bytes][type u8][value_len u32][value bytes]
  type 0 (UInt)     : value_len = 8,     value = uint64
  type 1 (Str)      : value_len = |str|, value = 原始字节
  type 2 (UIntArray): value_len = n×8,   value = n 个 uint64
```

- 空记录：`field_count = 0`（共 4 字节）。
- v1 **不支持** Record 字段（序列化时跳过不写）。
- 模型规格头（`model_serialization.hpp` 的 spec header）使用 v1，字节语义不变。

## 3. v2（地址布局）

```
[field_count u32]                       // 含哨兵在内
field[0] := 哨兵，按 v1 布局**内联**编码（v2 中唯一的内联字段）：
            [key_len=13]["kvrec_version"][type=0][value_len=8][u64 version=2]
field[i>0] := [key_len u32][key bytes][type u8][addr u64]   // 只记 key、类型、地址
数据块     := [value_len u64][value bytes]                  // 每块自带长度
  addr := 该值数据块在**文件内的绝对偏移**
```

### 3.1 写入方约定（addr 连续性）

数据块**按字段顺序、紧随字段表连续排布**。解析器逐字段校验
`addr == 期望游标`；整体搬迁（换基址复制字节）会**硬报错**，
而不是静默读到错值。因此：

- 嵌套 Record 的子记录以「自身数据块起点 + 8」为 base 递归序列化
  （其内部 addr 同样是文件绝对偏移）。
- 搬动一段 kvrec 字节（例如把它嵌进另一个文件）必须同时**重写所有 addr**。

### 3.2 大小计算

```
fields_blob_size(fs) = 4 + SENTINEL_INLINE_SIZE
                     + Σ_f (4 + |key_f| + 1 + 8)      // 字段头
                     + Σ_f (8 + value_size(f))         // 数据块（含长度前缀）
  SENTINEL_INLINE_SIZE = 4 + 13 + 1 + 4 + 8 = 30
  value_size(UInt) = 8; value_size(Str) = |s|;
  value_size(UIntArray) = n×8; value_size(Record) = fields_blob_size(children)
```

## 4. 版本自举（v1/v2 判别）

把 `field[0]` 按 **v1 布局**读出：

- `key == "kvrec_version"` ⇒ 本记录为 **v2**，余下字段按 v2 布局解析
  （哨兵必须是 UInt 且值 = 2，否则报错）；
- 否则 ⇒ 本记录为 **v1**（`field[0]` 已被正确按 v1 消费，继续 v1 解析）。

即：**无版本字段 = v1，有 = 按该版本读**。`"kvrec_version"` 是**保留键名**，
业务字段不得使用。空记录（`field_count = 0`）按 v1 处理。

未来 v3+：只需让哨兵值取 3…，解析器按值分派；布局不兼容时旧解析器
对未知版本号**报错**（不猜）。

## 5. 参考实现与合规测试

- **C++ 权威**：`KeyValueRecord::serialize / serialize_v2 / parse / parse_at`；
  单测 `src/kvrec_test.cpp`（v1 round-trip、版本自举、v2 round-trip、嵌套）。
- **Python 参考**：`scripts/kvrecord.py` —— `loads()/dumps()/load()/save()`，
  仅标准库。**合规口径**：对同一逻辑记录，Python `dumps_v2()` 输出必须与
  C++ `serialize_v2()` **逐字节一致**（主仓 `bpe_vocab_stories.nnvocab` 等
  真实产物可作对拍样本）。

```python
from kvrecord import KVRecord
r = KVRecord()
r.set_u64("vocab_size", 50257)
r.set_str("model", "bpe")
r.set_array("merges", [1, 2, 3])
blob = r.dumps_v2()          # 与 C++ serialize_v2() 逐字节一致
r2 = KVRecord.loads(blob)    # 版本自举：自动判别 v1/v2
assert r2.get_u64("vocab_size") == 50257
```
