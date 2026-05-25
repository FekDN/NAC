`include "nac_defs.vh"

module nac_mmap_engine #(
    parameter NUM_TICKS = 1024,
    parameter TICK_WIDTH = 10,
    parameter MAX_CMDS_PER_TICK = 4
) (
    input  wire clk,
    input  wire rst,

    input  wire cfg_we,
    input  wire [TICK_WIDTH-1:0] cfg_tick,
    input  wire [1:0] cfg_slot,
    input  wire cfg_valid,
    input  wire [7:0] cfg_action,
    input  wire [15:0] cfg_target,

    input  wire tick_valid,
    input  wire [TICK_WIDTH-1:0] tick_id,
    output reg  busy,

    output reg  preload_valid,
    output reg  [15:0] preload_target,

    output reg  free_valid,
    output reg  [15:0] free_target,

    output reg  save_result_valid,
    output reg  [15:0] save_result_src,
    output reg  [15:0] save_result_target,

    output reg  forward_valid,
    output reg  [15:0] forward_src,
    output reg  [15:0] forward_dst
);
    localparam TOTAL_SLOTS = NUM_TICKS * MAX_CMDS_PER_TICK;

    reg cmd_valid_ram [0:TOTAL_SLOTS-1];
    reg [7:0] action_ram [0:TOTAL_SLOTS-1];
    reg [15:0] target_ram [0:TOTAL_SLOTS-1];

    reg [TICK_WIDTH-1:0] active_tick;
    reg [1:0] slot;
    integer cfg_index;
    integer run_index;
    integer i;

    initial begin
        for (i = 0; i < TOTAL_SLOTS; i = i + 1) begin
            cmd_valid_ram[i] = 1'b0;
            action_ram[i] = 8'd0;
            target_ram[i] = 16'd0;
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            busy <= 1'b0;
            active_tick <= {TICK_WIDTH{1'b0}};
            slot <= 2'd0;
            preload_valid <= 1'b0;
            preload_target <= 16'd0;
            free_valid <= 1'b0;
            free_target <= 16'd0;
            save_result_valid <= 1'b0;
            save_result_src <= 16'd0;
            save_result_target <= 16'd0;
            forward_valid <= 1'b0;
            forward_src <= 16'd0;
            forward_dst <= 16'd0;
        end else begin
            preload_valid <= 1'b0;
            free_valid <= 1'b0;
            save_result_valid <= 1'b0;
            forward_valid <= 1'b0;

            if (cfg_we) begin
                cfg_index = (cfg_tick * MAX_CMDS_PER_TICK) + cfg_slot;
                cmd_valid_ram[cfg_index] <= cfg_valid;
                action_ram[cfg_index] <= cfg_action;
                target_ram[cfg_index] <= cfg_target;
            end

            if (!busy && tick_valid) begin
                active_tick <= tick_id;
                slot <= 2'd0;
                busy <= 1'b1;
            end else if (busy) begin
                run_index = (active_tick * MAX_CMDS_PER_TICK) + slot;
                if (cmd_valid_ram[run_index]) begin
                    case (action_ram[run_index])
                        `NAC_MMAP_PRELOAD: begin
                            preload_valid <= 1'b1;
                            preload_target <= target_ram[run_index];
                        end
                        `NAC_MMAP_FREE: begin
                            free_valid <= 1'b1;
                            free_target <= target_ram[run_index];
                        end
                        `NAC_MMAP_SAVE_RESULT: begin
                            save_result_valid <= 1'b1;
                            save_result_src <= active_tick;
                            save_result_target <= target_ram[run_index];
                        end
                        `NAC_MMAP_FORWARD: begin
                            forward_valid <= 1'b1;
                            forward_src <= active_tick;
                            forward_dst <= target_ram[run_index];
                        end
                        default: begin
                        end
                    endcase
                end

                if (slot == MAX_CMDS_PER_TICK - 1) begin
                    busy <= 1'b0;
                end else begin
                    slot <= slot + 2'd1;
                end
            end
        end
    end
endmodule
