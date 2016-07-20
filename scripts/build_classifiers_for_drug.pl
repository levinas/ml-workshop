#! /usr/bin/env perl

use strict;
use Carp;

my $usage = "Usage: $0 drug_NSC_id feature_data_file_list out_dir bydrug_zscore_dir scratch_dir\n\n";

# Example: $0 NSC_169780 feature_data_file_list /bigdata/fangfang/CellMiner.byDrug/runs /bigdata/fangfang/CellMiner.byDrug/byDrug.z

my $drug_file = shift @ARGV;
my $geno_file = shift @ARGV;
my $out_dir   = shift @ARGV;
my $z_dir     = shift @ARGV;
my $scratch   = shift @ARGV;

$drug_file && $geno_file or die $usage;

$z_dir   ||= 'byDrug.z';
$out_dir ||= 'runs';

-d $z_dir or die "Directory not found: $z_dir\n";

my $drug_dir = "$out_dir/$drug_file";
run("mkdir -p $drug_dir");

my @assays = map { chomp; $_ } `cut -f1 $geno_file`;

if ($scratch) {
    run("mkdir -p $scratch");
    my @new_assays;
    for my $f (@assays) {
        my $newf = $f; $newf =~ s/.*\///; $newf = join('/', $scratch, $newf);
        run("rsync -a \"$f\" \"$newf\"");
        push @new_assays, $newf;
    }
    @assays = @new_assays;
    # print join("\n", @assays) . "\n";
}

for my $assay (@assays) {
    my $base = $assay;
    $base =~ s/.*\///;
    $base =~ s/\.transposed.*//;
    $base =~ s/[()]/_/g;
    my $f1 = "$z_dir/$drug_file";
    my $f2 = "\"$assay\"";
    my $out = "$drug_dir/$drug_file.$base";

    run("join <(head -n 1 $f1) <(head -n 1 $f2) > $out\n");
    run("join <(tail -n +2 $f1|sort) <(tail -n +2 $f2|sort) >> $out\n");
    run("cd $drug_dir &&  ~/fs/ml-workshop/scripts/classify.py -m mean -f 5 $drug_file.$base 2>/dev/stdout |tee $drug_file.$base.log");
}


sub run { system("bash", "-c", @_) == 0 or confess("FAILED: ". join(" ", @_)); }
