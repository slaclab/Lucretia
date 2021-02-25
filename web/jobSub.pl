#!/usr/bin/perl

use Getopt::Std;

getopts('r');

$maxSeeds=100;

@seedList=(); @rerunList=();
if ($opt_r) {
  opendir(DATA, 'data') || die "Cannot open data directory: $!";
  @rerunFiles = (grep(/final/, (readdir DATA)));
  closedir(DATA);
  foreach (@rerunFiles) {
    m/_(\d+)/;
    push @rerunList, ($1);
  }
  @runningJobs = `bjobs -w | grep simLET`;
  for ($iSeed=1; $iSeed<=$maxSeeds; $iSeed++) {
    $isVeto=0;
    foreach (@rerunList) {
      $isVeto=1 if ($_ == $iSeed);
    }
    foreach (@runningJobs) {
      m/_(\d+)\.out/;
      $isVeto=1 if ($1 == $iSeed);
    }
    push @seedList, ($iSeed) if !($isVeto);
  }
}
else {
  @seedList=(1..100);
}
$isFirst=0; $lastLength=0;
foreach( @seedList ) {
`bsub -r -C 0 -q xlong \"simLET_run.csh $_ > jobOutputs/simLET_$_.out\"`;
  if ($isFirst) {
    for ($nDigits=1; $nDigits<=$lastLength; $nDigits++) {
      print "\b";
    }
    $lastLength=length;
    print "$_";
  }
  else {
    $isFirst=1;
    print "Submitting Job: $_";
    $lastLength=length;
  }
}
print "\n";
