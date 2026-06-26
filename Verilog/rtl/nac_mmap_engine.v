`include "nac_defs.vh"

module nac_mmap_engine #(
    parameter NUM_TICKS = 1024,
    parameter TICK_WIDTH = 10,
    parameter MAX_CMDS_PER_TICK = 256,
    parameter CMD_SLOT_WIDTH = 8
) (
    input  wire clk,
    input  wire rst,

    input  wire cfg_we,
    input  wire [TICK_WIDTH-1:0] cfg_tick,
    input  wire [CMD_SLOT_WIDTH-1:0] cfg_slot,
    input  wire cfg_valid,
    input  wire [7:0] cfg_action,
    input  wire [15:0] cfg_target,
    input  wire cfg_static,
    input  wire clear_static,

    input  wire tick_valid,
    input  wire [TICK_WIDTH-1:0] tick_id,
    output reg  busy,

    output reg  preload_valid,
    input  wire preload_ready,
    output reg  [15:0] preload_target,
    output reg  preload_static,

    output reg  free_valid,
    input  wire free_ready,
    output reg  [15:0] free_target,
    output reg  free_static,

    output reg  save_result_valid,
    input  wire save_result_ready,
    output reg  [15:0] save_result_src,
    output reg  [15:0] save_result_target,
    output reg  save_result_static,

    output reg  forward_valid,
    input  wire forward_ready,
    output reg  [15:0] forward_src,
    output reg  [15:0] forward_dst,
    output reg  forward_static,

    output reg  error
);
    localparam TOTAL_SLOTS = NUM_TICKS * MAX_CMDS_PER_TICK;

    reg cmd_valid_ram [0:TOTAL_SLOTS-1];
    reg [7:0] action_ram [0:TOTAL_SLOTS-1];
    reg [15:0] target_ram [0:TOTAL_SLOTS-1];
    reg static_ram [0:TOTAL_SLOTS-1];
    reg static_done_ram [0:TOTAL_SLOTS-1];

    reg [TICK_WIDTH-1:0] active_tick;
    reg [TICK_WIDTH-1:0] last_tick;
    reg [TICK_WIDTH-1:0] pending_tick;
    reg last_valid;
    reg pending_valid;
    reg [CMD_SLOT_WIDTH-1:0] slot;

    reg cmd_pending;
    reg [7:0] pending_action;
    reg pending_static;
    reg [31:0] pending_index;

    integer cfg_index;
    integer run_index;
    integer i;

    wire pending_ready =
        (pending_action == `NAC_MMAP_PRELOAD)     ? preload_ready :
        (pending_action == `NAC_MMAP_FREE)        ? free_ready :
        (pending_action == `NAC_MMAP_SAVE_RESULT) ? save_result_ready :
        (pending_action == `NAC_MMAP_FORWARD)     ? forward_ready :
                                                     1'b1;

    wire cmd_accept = cmd_pending && pending_ready;

    function is_valid_action;
        input [7:0] action;
        begin
            is_valid_action = (action == `NAC_MMAP_PRELOAD) ||
                              (action == `NAC_MMAP_FREE) ||
                              (action == `NAC_MMAP_SAVE_RESULT) ||
                              (action == `NAC_MMAP_FORWARD);
        end
    endfunction

    initial begin
        for (i = 0; i < TOTAL_SLOTS; i = i + 1) begin
            cmd_valid_ram[i] = 1'b0;
            action_ram[i] = 8'd0;
            target_ram[i] = 16'd0;
            static_ram[i] = 1'b0;
            static_done_ram[i] = 1'b0;
        end
    end

    task clear_outputs;
        begin
            preload_valid <= 1'b0;
            free_valid <= 1'b0;
            save_result_valid <= 1'b0;
            forward_valid <= 1'b0;
            preload_static <= 1'b0;
            free_static <= 1'b0;
            save_result_static <= 1'b0;
            forward_static <= 1'b0;
        end
    endtask

    task advance_slot_or_tick;
        reg [TICK_WIDTH-1:0] range_end_tick;
        begin
            range_end_tick = pending_tick;
            if (tick_valid && tick_id > pending_tick) begin
                range_end_tick = tick_id;
            end

            if (slot == (MAX_CMDS_PER_TICK - 1)) begin
                last_tick <= active_tick;
                last_valid <= 1'b1;
                if (active_tick == range_end_tick) begin
                    busy <= 1'b0;
                    pending_valid <= 1'b0;
                    slot <= {CMD_SLOT_WIDTH{1'b0}};
                end else begin
                    pending_tick <= range_end_tick;
                    pending_valid <= 1'b1;
                    active_tick <= active_tick + {{(TICK_WIDTH-1){1'b0}}, 1'b1};
                    slot <= {CMD_SLOT_WIDTH{1'b0}};
                end
            end else begin
                slot <= slot + {{(CMD_SLOT_WIDTH-1){1'b0}}, 1'b1};
            end
        end
    endtask

    task start_pending_range;
        input [TICK_WIDTH-1:0] newest_tick;
        begin
            pending_tick <= newest_tick;
            pending_valid <= 1'b1;
            if (last_valid && newest_tick > last_tick) begin
                active_tick <= last_tick + {{(TICK_WIDTH-1){1'b0}}, 1'b1};
            end else begin
                active_tick <= {TICK_WIDTH{1'b0}};
                last_valid <= 1'b0;
            end
            slot <= {CMD_SLOT_WIDTH{1'b0}};
            busy <= 1'b1;
        end
    endtask

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            active_tick <= {TICK_WIDTH{1'b0}};
            last_tick <= {TICK_WIDTH{1'b0}};
            pending_tick <= {TICK_WIDTH{1'b0}};
            last_valid <= 1'b0;
            pending_valid <= 1'b0;
            slot <= {CMD_SLOT_WIDTH{1'b0}};
            cmd_pending <= 1'b0;
            pending_action <= 8'd0;
            pending_static <= 1'b0;
            pending_index <= 32'd0;
            preload_valid <= 1'b0;
            preload_target <= 16'd0;
            preload_static <= 1'b0;
            free_valid <= 1'b0;
            free_target <= 16'd0;
            free_static <= 1'b0;
            save_result_valid <= 1'b0;
            save_result_src <= 16'd0;
            save_result_target <= 16'd0;
            save_result_static <= 1'b0;
            forward_valid <= 1'b0;
            forward_src <= 16'd0;
            forward_dst <= 16'd0;
            forward_static <= 1'b0;
            error <= 1'b0;
        end else begin
            if (clear_static) begin
                for (i = 0; i < TOTAL_SLOTS; i = i + 1) begin
                    static_done_ram[i] <= 1'b0;
                end
            end

            if (cfg_we) begin
                if (cfg_slot >= MAX_CMDS_PER_TICK || (cfg_valid && !is_valid_action(cfg_action))) begin
                    error <= 1'b1;
                end else begin
                    cfg_index = (cfg_tick * MAX_CMDS_PER_TICK) + cfg_slot;
                    cmd_valid_ram[cfg_index] <= cfg_valid;
                    action_ram[cfg_index] <= cfg_action;
                    target_ram[cfg_index] <= cfg_target;
                    static_ram[cfg_index] <= cfg_static;
                    if (!cfg_valid) static_done_ram[cfg_index] <= 1'b0;
                end
            end

            if (tick_valid) begin
                if (busy && last_valid && tick_id <= last_tick) begin
                    error <= 1'b1;
                end else begin
                    pending_valid <= 1'b1;
                    if (!pending_valid || tick_id > pending_tick || (last_valid && tick_id <= last_tick)) begin
                        pending_tick <= tick_id;
                    end
                end
            end

            if (cmd_accept) begin
                cmd_pending <= 1'b0;
                if (pending_static && pending_action == `NAC_MMAP_PRELOAD) begin
                    static_done_ram[pending_index] <= 1'b1;
                end
                pending_action <= 8'd0;
                pending_static <= 1'b0;
                clear_outputs();
                advance_slot_or_tick();
            end else if (!cmd_pending) begin
                clear_outputs();

                if (!busy) begin
                    if (tick_valid) begin
                        start_pending_range(tick_id);
                    end else if (pending_valid) begin
                        start_pending_range(pending_tick);
                    end
                end else begin
                    run_index = (active_tick * MAX_CMDS_PER_TICK) + slot;
                    if (cmd_valid_ram[run_index] &&
                        !(static_ram[run_index] && action_ram[run_index] == `NAC_MMAP_PRELOAD && static_done_ram[run_index]) &&
                        !(static_ram[run_index] && action_ram[run_index] == `NAC_MMAP_FREE)) begin
                        pending_action <= action_ram[run_index];
                        pending_static <= static_ram[run_index];
                        pending_index <= run_index;
                        cmd_pending <= 1'b1;
                        case (action_ram[run_index])
                            `NAC_MMAP_PRELOAD: begin
                                preload_valid <= 1'b1;
                                preload_target <= target_ram[run_index];
                                preload_static <= static_ram[run_index];
                            end
                            `NAC_MMAP_FREE: begin
                                free_valid <= 1'b1;
                                free_target <= target_ram[run_index];
                                free_static <= static_ram[run_index];
                            end
                            `NAC_MMAP_SAVE_RESULT: begin
                                save_result_valid <= 1'b1;
                                save_result_src <= active_tick;
                                save_result_target <= target_ram[run_index];
                                save_result_static <= static_ram[run_index];
                            end
                            `NAC_MMAP_FORWARD: begin
                                forward_valid <= 1'b1;
                                forward_src <= active_tick;
                                forward_dst <= target_ram[run_index];
                                forward_static <= static_ram[run_index];
                            end
                            default: begin
                                error <= 1'b1;
                                cmd_pending <= 1'b0;
                                advance_slot_or_tick();
                            end
                        endcase
                    end else begin
                        advance_slot_or_tick();
                    end
                end
            end
        end
    end
endmodule
