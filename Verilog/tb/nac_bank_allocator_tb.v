`timescale 1ns/1ps

module nac_bank_allocator_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    reg rst;
    reg alloc_req;
    wire alloc_valid;
    wire [1:0] alloc_bank;
    wire alloc_fail;
    reg free_req;
    reg [1:0] free_bank;
    wire free_error;

    nac_bank_allocator #(
        .BANKS(3),
        .BANK_BITS(2)
    ) dut (
        .clk(clk),
        .rst(rst),
        .alloc_req(alloc_req),
        .alloc_valid(alloc_valid),
        .alloc_bank(alloc_bank),
        .alloc_fail(alloc_fail),
        .free_req(free_req),
        .free_bank(free_bank),
        .free_error(free_error)
    );

    task request_alloc;
        input [1:0] expected_bank;
        begin
            @(negedge clk);
            alloc_req = 1'b1;
            @(negedge clk);
            alloc_req = 1'b0;
            if (!alloc_valid || alloc_bank != expected_bank || alloc_fail) begin
                $fatal(1, "allocation failed expected=%0d valid=%0b bank=%0d fail=%0b",
                       expected_bank, alloc_valid, alloc_bank, alloc_fail);
            end
        end
    endtask

    task request_free;
        input [1:0] bank;
        input expect_error;
        begin
            @(negedge clk);
            free_bank = bank;
            free_req = 1'b1;
            @(negedge clk);
            free_req = 1'b0;
            if (free_error !== expect_error) begin
                $fatal(1, "free status failed bank=%0d expected_error=%0b actual=%0b",
                       bank, expect_error, free_error);
            end
        end
    endtask

    initial begin
        rst = 1'b1;
        alloc_req = 1'b0;
        free_req = 1'b0;
        free_bank = 2'd0;

        repeat (3) @(posedge clk);
        rst = 1'b0;

        request_alloc(2'd0);
        request_alloc(2'd1);
        request_alloc(2'd2);

        @(negedge clk);
        alloc_req = 1'b1;
        @(negedge clk);
        alloc_req = 1'b0;
        if (!alloc_fail || alloc_valid) $fatal(1, "allocator did not report full condition");

        request_free(2'd1, 1'b0);
        request_free(2'd1, 1'b1);
        request_free(2'd3, 1'b1);
        request_alloc(2'd1);

        $display("nac_bank_allocator_tb PASS");
        $finish;
    end
endmodule
