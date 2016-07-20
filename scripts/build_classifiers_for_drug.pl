#! /usr/bin/env perl

use strict;
use Carp;

my $usage = "Usage: $0 drug_NSC_id feature_data_file_list\n\n";

# Example: $0 NSC_169780 feature_data_file_list

my $drug_file = shift @ARGV;
my $geno_file = shift @ARGV;

$drug_file && $geno_File or die $usage;

my $runs_dir = "/bigdata/fangfang/CellMiner.byDrug/runs";

chdir($runs_dir);
run("mkdir -p $drug_file");



sub run { system(@_) == 0 or confess("FAILED: ". join(" ", @_)); }
