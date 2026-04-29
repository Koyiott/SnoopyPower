
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.math_real.all;
package tdc_pack is
    function sel_width(count : positive) return positive;
    function fine_width(count : positive) return positive;
    function coarse_width(count : positive) return positive;
    function state_width(depth : positive) return positive;
    function weight_width(depth : positive) return positive;

    constant bits_per_fine_c : positive := 4;
    constant bits_per_coarse_c : positive := 2;
    constant bits_per_depth_c : positive := 4;
end tdc_pack;

package body tdc_pack is
    function sel_width(count : positive) return positive is
    begin
        return integer(ceil(log2(real(count))));
    end function;

    function fine_width(count : positive) return positive is
    begin
        return bits_per_fine_c * count;
    end function;

    function coarse_width(count : positive) return positive is
    begin
        return bits_per_coarse_c * count;
    end function;

    function state_width(depth : positive) return positive is -- Total bits required to store the full state of the TDC
    begin
        return bits_per_depth_c * depth; -- One stage is composed of 4 LUTs and 4 FFs
    end function;

    function weight_width(depth : positive) return positive is
    begin
        return integer(ceil(log2(real(bits_per_depth_c * depth)))); -- Weights value
    end function;
end tdc_pack;