// End-to-end inference phase controller.
//
// This block keeps TISA preprocessing/postprocessing outside the OPS graph
// while still running it as part of the FPGA lifecycle:
//   optional PRE TISA resources -> PRE TISA -> OPS graph -> optional POST TISA
//   resources -> POST TISA.
//
// Resource requests use existing NAC section offsets. PROC is the tokenizer
// manifest section, RSRC contains embedded tokenizer resources. No new NAC,
// ABCD, MEP, or TISA opcodes are introduced here.
module nac_model_lifecycle #(
    parameter PROC_SECTION_INDEX = 4'd6,
    parameter RSRC_SECTION_INDEX = 4'd9
) (
    input  wire clk,
    input  wire rst,
    input  wire start,
    input  wire model_configured,

    input  wire [11*64-1:0] section_offsets,
    input  wire [63:0] nac_file_size,

    input  wire pre_tisa_enable,
    input  wire post_tisa_enable,
    input  wire [1:0] pre_load_mask,
    input  wire [1:0] pre_free_mask,
    input  wire [1:0] post_load_mask,
    input  wire [1:0] post_free_mask,

    output reg  resource_req_valid,
    input  wire resource_req_ready,
    output reg  resource_req_is_free,
    output reg  [1:0] resource_req_phase,
    output reg  [1:0] resource_req_kind,
    output reg  [3:0] resource_req_section_index,
    output reg  [63:0] resource_req_offset,
    output reg  [63:0] resource_req_size,
    input  wire resource_done,
    input  wire resource_error,

    output reg  pre_tisa_start,
    input  wire pre_tisa_done,
    input  wire pre_tisa_error,

    output reg  model_run_start,
    input  wire model_run_done,
    input  wire model_run_error,

    output reg  post_tisa_start,
    input  wire post_tisa_done,
    input  wire post_tisa_error,

    output wire busy,
    output reg  done,
    output reg  error,
    output reg  [3:0] state_debug
);
    localparam RES_PROC = 2'd0;
    localparam RES_RSRC = 2'd1;
    localparam PHASE_PRE  = 2'd0;
    localparam PHASE_POST = 2'd1;

    localparam S_IDLE       = 4'd0;
    localparam S_PHASE_LOAD = 4'd1;
    localparam S_RES_REQ    = 4'd2;
    localparam S_RES_WAIT   = 4'd3;
    localparam S_TISA_START = 4'd4;
    localparam S_TISA_WAIT  = 4'd5;
    localparam S_PHASE_FREE = 4'd6;
    localparam S_MODEL_ST   = 4'd7;
    localparam S_MODEL_WAIT = 4'd8;
    localparam S_DONE       = 4'd9;
    localparam S_ERROR      = 4'd10;

    reg [3:0] state;
    reg [1:0] phase;
    reg [1:0] pending_load_mask;
    reg [1:0] pending_free_mask;
    reg [1:0] loaded_mask;
    reg [1:0] configured_free_mask;
    reg [1:0] active_kind;
    reg active_is_free;
    reg cleanup_to_error;
    reg go_model_after_free;
    reg go_post_after_model;

    assign busy = (state != S_IDLE) && (state != S_DONE) && (state != S_ERROR);

    function [63:0] section_offset;
        input [3:0] idx;
        begin
            section_offset = section_offsets[idx*64 +: 64];
        end
    endfunction

    function [3:0] section_index_for_kind;
        input [1:0] kind;
        begin
            case (kind)
                RES_PROC: section_index_for_kind = PROC_SECTION_INDEX;
                RES_RSRC: section_index_for_kind = RSRC_SECTION_INDEX;
                default: section_index_for_kind = 4'd0;
            endcase
        end
    endfunction

    function [63:0] section_size;
        input [3:0] idx;
        reg [63:0] base;
        reg [63:0] next_off;
        reg [63:0] candidate;
        integer i;
        begin
            base = section_offset(idx);
            next_off = nac_file_size;
            for (i = 0; i < 11; i = i + 1) begin
                candidate = section_offset(i[3:0]);
                if (candidate > base && (next_off == 64'd0 || candidate < next_off))
                    next_off = candidate;
            end
            if (base != 64'd0 && next_off > base)
                section_size = next_off - base;
            else
                section_size = 64'd0;
        end
    endfunction

    function [1:0] first_kind;
        input [1:0] mask;
        begin
            if (mask[RES_PROC])
                first_kind = RES_PROC;
            else
                first_kind = RES_RSRC;
        end
    endfunction

    task setup_phase;
        input [1:0] new_phase;
        input [1:0] load_mask;
        input [1:0] free_mask;
        input next_is_model;
        begin
            phase <= new_phase;
            pending_load_mask <= load_mask;
            pending_free_mask <= 2'd0;
            loaded_mask <= 2'd0;
            configured_free_mask <= free_mask;
            cleanup_to_error <= 1'b0;
            go_model_after_free <= next_is_model;
            if (load_mask != 2'd0)
                state <= S_PHASE_LOAD;
            else
                state <= S_TISA_START;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            state <= S_IDLE;
            phase <= PHASE_PRE;
            pending_load_mask <= 2'd0;
            pending_free_mask <= 2'd0;
            loaded_mask <= 2'd0;
            configured_free_mask <= 2'd0;
            active_kind <= RES_PROC;
            active_is_free <= 1'b0;
            cleanup_to_error <= 1'b0;
            go_model_after_free <= 1'b0;
            go_post_after_model <= 1'b0;
            resource_req_valid <= 1'b0;
            resource_req_is_free <= 1'b0;
            resource_req_phase <= PHASE_PRE;
            resource_req_kind <= RES_PROC;
            resource_req_section_index <= 4'd0;
            resource_req_offset <= 64'd0;
            resource_req_size <= 64'd0;
            pre_tisa_start <= 1'b0;
            model_run_start <= 1'b0;
            post_tisa_start <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            state_debug <= S_IDLE;
        end else begin
            pre_tisa_start <= 1'b0;
            model_run_start <= 1'b0;
            post_tisa_start <= 1'b0;
            state_debug <= state;

            case (state)
                S_IDLE: begin
                    done <= 1'b0;
                    error <= 1'b0;
                    resource_req_valid <= 1'b0;
                    if (start) begin
                        if (!model_configured) begin
                            error <= 1'b1;
                            state <= S_ERROR;
                        end else if (pre_tisa_enable) begin
                            go_post_after_model <= post_tisa_enable;
                            setup_phase(PHASE_PRE, pre_load_mask, pre_free_mask, 1'b1);
                        end else begin
                            go_post_after_model <= post_tisa_enable;
                            state <= S_MODEL_ST;
                        end
                    end
                end

                S_PHASE_LOAD: begin
                    if (pending_load_mask == 2'd0) begin
                        state <= S_TISA_START;
                    end else begin
                        active_kind <= first_kind(pending_load_mask);
                        active_is_free <= 1'b0;
                        state <= S_RES_REQ;
                    end
                end

                S_PHASE_FREE: begin
                    if (pending_free_mask == 2'd0) begin
                        if (cleanup_to_error) begin
                            error <= 1'b1;
                            state <= S_ERROR;
                        end else if (go_model_after_free) begin
                            state <= S_MODEL_ST;
                        end else begin
                            done <= 1'b1;
                            state <= S_DONE;
                        end
                    end else begin
                        active_kind <= first_kind(pending_free_mask);
                        active_is_free <= 1'b1;
                        state <= S_RES_REQ;
                    end
                end

                S_RES_REQ: begin
                    if (!resource_req_valid &&
                        (section_offset(section_index_for_kind(active_kind)) == 64'd0 ||
                         (!active_is_free && section_size(section_index_for_kind(active_kind)) == 64'd0))) begin
                        resource_req_valid <= 1'b0;
                        cleanup_to_error <= 1'b1;
                        pending_free_mask <= loaded_mask & configured_free_mask;
                        state <= (loaded_mask & configured_free_mask) ? S_PHASE_FREE : S_ERROR;
                        error <= ~(|(loaded_mask & configured_free_mask));
                    end else if (!resource_req_valid) begin
                        resource_req_valid <= 1'b1;
                        resource_req_is_free <= active_is_free;
                        resource_req_phase <= phase;
                        resource_req_kind <= active_kind;
                        resource_req_section_index <= section_index_for_kind(active_kind);
                        resource_req_offset <= section_offset(section_index_for_kind(active_kind));
                        resource_req_size <= section_size(section_index_for_kind(active_kind));
                    end else if (resource_req_ready) begin
                        resource_req_valid <= 1'b0;
                        state <= S_RES_WAIT;
                    end
                end

                S_RES_WAIT: begin
                    if (resource_error) begin
                        cleanup_to_error <= 1'b1;
                        pending_free_mask <= loaded_mask & configured_free_mask;
                        if (loaded_mask & configured_free_mask) begin
                            state <= S_PHASE_FREE;
                        end else begin
                            error <= 1'b1;
                            state <= S_ERROR;
                        end
                    end else if (resource_done) begin
                        if (active_is_free) begin
                            pending_free_mask[active_kind] <= 1'b0;
                            loaded_mask[active_kind] <= 1'b0;
                            state <= S_PHASE_FREE;
                        end else begin
                            pending_load_mask[active_kind] <= 1'b0;
                            loaded_mask[active_kind] <= 1'b1;
                            state <= S_PHASE_LOAD;
                        end
                    end
                end

                S_TISA_START: begin
                    if (phase == PHASE_PRE)
                        pre_tisa_start <= 1'b1;
                    else
                        post_tisa_start <= 1'b1;
                    state <= S_TISA_WAIT;
                end

                S_TISA_WAIT: begin
                    if ((phase == PHASE_PRE && pre_tisa_error) ||
                        (phase == PHASE_POST && post_tisa_error)) begin
                        cleanup_to_error <= 1'b1;
                        pending_free_mask <= loaded_mask & configured_free_mask;
                        if (loaded_mask & configured_free_mask)
                            state <= S_PHASE_FREE;
                        else begin
                            error <= 1'b1;
                            state <= S_ERROR;
                        end
                    end else if ((phase == PHASE_PRE && pre_tisa_done) ||
                                 (phase == PHASE_POST && post_tisa_done)) begin
                        cleanup_to_error <= 1'b0;
                        pending_free_mask <= loaded_mask & configured_free_mask;
                        state <= S_PHASE_FREE;
                    end
                end

                S_MODEL_ST: begin
                    model_run_start <= 1'b1;
                    state <= S_MODEL_WAIT;
                end

                S_MODEL_WAIT: begin
                    if (model_run_error) begin
                        error <= 1'b1;
                        state <= S_ERROR;
                    end else if (model_run_done) begin
                        if (go_post_after_model && post_tisa_enable) begin
                            setup_phase(PHASE_POST, post_load_mask, post_free_mask, 1'b0);
                        end else begin
                            done <= 1'b1;
                            state <= S_DONE;
                        end
                    end
                end

                S_DONE: begin
                    if (start) begin
                        done <= 1'b0;
                        error <= 1'b0;
                        if (!model_configured) begin
                            error <= 1'b1;
                            state <= S_ERROR;
                        end else if (pre_tisa_enable) begin
                            go_post_after_model <= post_tisa_enable;
                            setup_phase(PHASE_PRE, pre_load_mask, pre_free_mask, 1'b1);
                        end else begin
                            go_post_after_model <= post_tisa_enable;
                            state <= S_MODEL_ST;
                        end
                    end
                end

                S_ERROR: begin
                    if (start) begin
                        error <= 1'b0;
                        done <= 1'b0;
                        if (!model_configured) begin
                            error <= 1'b1;
                        end else if (pre_tisa_enable) begin
                            go_post_after_model <= post_tisa_enable;
                            setup_phase(PHASE_PRE, pre_load_mask, pre_free_mask, 1'b1);
                        end else begin
                            go_post_after_model <= post_tisa_enable;
                            state <= S_MODEL_ST;
                        end
                    end
                end

                default: begin
                    error <= 1'b1;
                    state <= S_ERROR;
                end
            endcase
        end
    end
endmodule
