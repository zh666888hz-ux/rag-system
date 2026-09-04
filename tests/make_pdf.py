"""
测试辅助：程序化生成最小合法 PDF。

为什么需要它：pypdf 只能「读」文本层，不能方便地「写」带文本的 PDF，
因此测试里我们手工构造一个符合 PDF 规范的最小文件，用来验证 loader 的文本抽取逻辑，
避免引入 reportlab 等额外重量级依赖。

中文文本如何被正确抽取：
1. PDF 内容流里的 Tj 文本用的是「字形码」，解析器要通过字体对象的 /ToUnicode
   CMap 把字形码映射回 Unicode 码点，才能还原出可读文本；
2. 因此这里把每个字符编码为单字节码位（0x01 起），写入内容流；
   再为字体挂一张 ToUnicode CMap（<字形码> <Unicode十六进制> 的 bfchar 表）；
3. pypdf 解析时读到 /ToUnicode 即可把码位还原成原始中文。

对象布局：
  1: 目录 Catalog      2: 页面树 Pages
  3..3+n-1: 页面对象（引用内容流与共享字体）
  3+n..3+2n-1: 各页内容流
  3+2n: 字体对象（含 /ToUnicode 引用）
  3+2n+1: ToUnicode CMap 流
"""
from __future__ import annotations


def _encode_text(text: str, char_to_code: dict[str, int]) -> bytes:
    """按给定码位表把文本编码为内容流字形码序列。

    调用方必须先在全部文本上建立统一的 char_to_code（见 _build_code_table），
    保证同一字符在所有页面中使用相同码位，否则跨页会错乱。
    """
    return bytes(char_to_code[ch] for ch in text)


def _build_code_table(texts: list[str]) -> dict[str, int]:
    """在全部页面文本上建立统一的「字符 → 码位」表（从 0x01 起）。"""
    table: dict[str, int] = {}
    for text in texts:
        for ch in text:
            if ch not in table:
                table[ch] = len(table) + 1
    return table


def _to_unicode_cmap(char_to_code: dict[str, int]) -> bytes:
    """生成把字形码映射回 Unicode 的 ToUnicode CMap 流。"""
    entries = [f"<{code:02X}> <{ord(ch):04X}>".encode() for ch, code in char_to_code.items()]
    return (
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        b"/CMapName /Adobe-Identity-UCS def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
        + f"{len(entries)} beginbfchar\n".encode()
        + b"\n".join(entries)
        + b"\nendbfchar\n"
        b"endcmap\n"
        b"end\n"
        b"end\n"
    )


def build_pdf(pages: list[str]) -> bytes:
    """生成一页或多页的合法 PDF 字节流（支持中文）。

    Args:
        pages: 每页的文本内容

    Returns:
        完整的 PDF 二进制内容
    """
    n = len(pages)

    # ---------- 依次构造各对象体 ----------
    bodies: list[bytes] = []

    # 1 号对象：目录
    bodies.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    # 2 号对象：页面树（引用各页面对象）
    kids = " ".join(f"{3 + i} 0 R" for i in range(n))
    bodies.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode("latin-1"))

    # 3 .. 3+n-1 号对象：页面对象（引用内容流与共享字体）
    for i in range(n):
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {3 + n + i} 0 R "
            f"/Resources << /Font << /F1 {3 + 2 * n} 0 R >> >> >>"
        )
        bodies.append(page_obj.encode("latin-1"))

    # 全局统一码位表：同一字符在所有页面使用相同字形码
    char_to_code = _build_code_table(pages)

    # 3+n .. 3+2n-1 号对象：内容流（写入文本绘制指令）
    for text in pages:
        content_codes = _encode_text(text, char_to_code)
        # 字形码以十六进制串 <..> 写入（纯 ASCII，可用 latin-1 编码）
        hex_str = content_codes.hex().upper()
        stream = f"BT /F1 12 Tf 72 720 Td <{hex_str}> Tj ET".encode("latin-1")
        body = (
            b"<< /Length " + str(len(stream)).encode() + b" >>\n"
            b"stream\n" + stream + b"\nendstream"
        )
        bodies.append(body)

    # 3+2n 号对象：共享内置字体（含 /ToUnicode 引用，指向下一个对象）
    cmap_obj_num = 3 + 2 * n + 1
    bodies.append(
        (
            f"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica "
            f"/ToUnicode {cmap_obj_num} 0 R >>"
        ).encode("latin-1")
    )

    # 3+2n+1 号对象：ToUnicode CMap 流（把所有码位映射回 Unicode）
    cmap_stream = _to_unicode_cmap(char_to_code)
    cmap_body = (
        b"<< /Length " + str(len(cmap_stream)).encode() + b" >>\n"
        b"stream\n" + cmap_stream + b"\nendstream"
    )
    bodies.append(cmap_body)

    # ---------- 组装文件 + 生成 xref 偏移表 ----------
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]  # offsets[i] = 对象 (i+1) 的起始字节偏移
    for num, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode("latin-1")
        out += body
        out += b"\nendobj\n"

    xref_pos = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode("latin-1")
    out += b"0000000000 65535 f \n"  # 自由对象条目
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode("latin-1")

    out += (
        f"trailer\n<< /Size {len(bodies) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("latin-1")
    return bytes(out)
