#! /usr/bin/env perl

use strict;

use Getopt::Long;

my $usage = "Usage: $0 cellminer_input.txt\n\n";

my ($help, $sr, $sc);

GetOptions("h|help"            => \$help,
           "sc|start_columns=i" => \$sc,
           "sr|start_rows=i"    => \$sr,
	  ) or die("Error in command line arguments\n");

$help and die $usage;

my $input = shift @ARGV or die $usage;

($sr, $sc) = guess_start_cell($input);

print STDERR "Transposing $input. Start cell = ($sr, $sc). Skipping rows 1..$sr and columns 2..$sc ...\n";

my @cell_lines;
my @ids;
my @data;

open(F, "<$input") or die "Could not open $input";
my $ln = 0;
while (<F>) {
    next if $ln++ < $sr;
    chomp;
    s/\s+$//;
    my @cols = split/\t/;
    last unless @cols;
    if ($ln == $sr+1) {
        @cell_lines = @cols[$sc..$#cols];
    } else {
        push @ids, $cols[0];
        for (my $i = $sc; $i < @cols; $i++) {
            push @{$data[$i-$sc]}, $cols[$i];
        }
    }
    # last if $ln > 100;
}
close(F);

print join("\t", 'CellLine', @ids) . "\n";
for (@data) {
    my $cl = shift @cell_lines;
    $cl =~ s/:/\./;
    print join("\t", $cl, @$_) . "\n";
}

sub guess_start_cell {
    my ($file) = @_;
    open(F, "<$file") or die "Could not open $file";
    my ($row, $col) = 0;
    my $line;
    while (<F>) {
        if (/BR[:.]MCF7/) {
            $line = $_;
            last;
        }
        $row++;
    }
    if ($line) {
        my @cols = split(/\t/, $line);
        for (@cols) {
            last if /BR[:.]MCF7/;
            $col++;
        }
    } else {
        close(F);
        return undef;
    }
    close(F);
    return ($row, $col);
}
