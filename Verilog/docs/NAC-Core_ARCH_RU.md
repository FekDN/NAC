# NAC-Core: соответствие стандартам NAC / MEP / TISA

## Что реализовано

`NAC-Core` - это аппаратная оболочка soft-core процессора для NAC v1.8 с
разделением Access/Execute:

- `nac_abcd_decoder` последовательно декодирует `OPS` как ABCD-инструкции.
- `nac_orch_fsm` является главным диспетчером: он не считает математику, а
  выдает команды tensor datapath, descriptor table и input/parameter provider.
- `nac_mmap_engine` является независимым MMAP-потоком и исполняет
  `PRELOAD`, `FREE`, `SAVE_RESULT`, `FORWARD` по tick-ам инструкции.
- `nac_dsp_pipeline` реализует переиспользуемый SIMD/DSP-тракт:
  MAC-массив, SFU lanes и аппаратные редукции sum/max.
- `nac_kernel_sequencer` разворачивает сложные классы `Softmax` и `LayerNorm`
  в несколько проходов одного и того же DSP-конвейера.
- `nac_scratchpad` и `nac_bank_allocator` дают основу для banked BRAM памяти.

## NAC

NAC v1.8 отражен так:

- Header: `nac_header_parser` проверяет `NAC\x02`, flags, IO counts, `d_model`
  и 11 section offsets.
- `PERM`: в железо грузится не строка сигнатуры, а компактные признаки
  `arity` и `needs_consts`; этого достаточно для чтения `C` и `D`.
- `CMAP`: строковые имена не сравниваются в логике. Runtime/loader заранее
  классифицирует их в `kernel_class`.
- `OPS`: `<INPUT>` и `<OUTPUT>` обрабатываются по special-правилам `A < 10`;
  обычные операции `A >= 10` уходят в kernel sideband вместе с `A/B/C/D`.
- `MMAP`: schedule хранится как tick table с несколькими командами на tick.

## MEP / ORCH

MEP рассматривается как верхний слой оркестрации. В RTL есть
`nac_mep_packetizer`, который сохраняет точные границы MEP-инструкций,
включая инструкции с `count`-зависимой длиной. Он не добавляет частных
macro-opcodes и не меняет формат MEP.

## TISA

`nac_tisa_packetizer` валидирует `TISA`, версию и точные
`opcode/payload_len/payload` границы. Payload bytes сохраняются без изменения;
частные правила под модели не вводятся.

## Time-multiplexing

Вместо отдельных блоков под каждый слой используется один физический тракт:

- MatMul / Linear / Conv2D: `NAC_DSP_MAC`.
- Elementwise ops: `ADD`, `SUB`, `MUL`, comparisons, `NEG`.
- Softmax: `REDUCE_MAX -> SUB -> EXP -> REDUCE_SUM -> SCALE`.
- LayerNorm: `REDUCE_SUM -> SUB -> MUL -> REDUCE_SUM -> RSQRT -> SCALE`.

Адресные генераторы и scratchpad banks должны подавать тайлы так, чтобы
`nac_dsp_pipeline` принимал новый вектор каждый такт после заполнения конвейера.
