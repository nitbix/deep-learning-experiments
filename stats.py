#!/usr/bin/python

import re
import yaml
import math

from stats_common import *

results_table="armsprop"
results_db="amosca02"                                                                                                                                                                            
results_host="localhost"  

methods, datasets, stats, stats_hist = extract_results(results_host, results_db, results_table)
hsep = " & "
vsep = " \\\\"
def make_line(first,mean,stdev,min_bold = False):
    it = []
    for i,x in enumerate(mean):
        if isinstance(x,float):
            it.append("{0:.2f}".format(x))
        else:
            it.append(str(x))
    mean = it
    if min_bold:
        str_mean = []
        for x in mean:
            if x == min(mean):
                str_mean.append("$\\mathbf{{ {0} }} $".format(x))
            else:
                    str_mean.append("$ {0} $".format(x))
    else:
        if len(stdev) > 0:
            str_mean = [
                    "$ {0}\% $ $ ({1:.2f})$".format(x,stdev[i])
                    for i,x in enumerate(mean)
            ]
        else:
            str_mean = ["$ {0} $".format(x) for x in mean]
    if first is not None:
        return first + hsep + hsep.join(str_mean) + vsep
    else:
        return hsep.join(str_mean) + vsep

def arrify(x,v):
    return [x['test'][v],x['valid'][v],x['epoch'][v]]
#MIDDLE TABLES
for dataset in sorted(datasets):
    print "\n"
    print dataset
    print """
\\begin{table}[h]
\\centering
\\begin{tabular}
    """
    print make_line("",
            ["Mean Test Acc (std)", "Mean Best Valid Acc (std)", "Epochs"],
            [])
    print "\\hline"
    line = []
    for method in sorted(methods):
        if dataset in stats and method in stats[dataset]:
            s = stats[dataset][method]
            print make_line(method,arrify(s,'mean'),arrify(s,'std'))
        else:
            print "missing \\"
    print """
\\hline
\\end{tabular}
\\end{table}
    """

#FINAL TABLE

def make_final_header(first,titles):
    it = []
    if first is not None:
        return first + hsep + hsep.join(titles) + vsep
    else:
        return hsep.join(str_mean) + vsep

print """
\\begin{table}[h]
\\centering
\\begin{tabular}{ l | c|c|c|c | c|c|c|c }
\\hline
 & \\multicolumn{4}{|c|}{accuracy ( \\%)} & \\multicolumn{4}{|c}{best epoch} \\
 & $\\mu$ & $\\sigma$ & min & max & $\\mu$ & $\\sigma$ & min & max \\
\\hline
"""

ordered_stats = ["mean", "std", "min", "max"]
for dataset in sorted(datasets):
    print "\\hline"
    print "\\multicolumn{9}{c}{" + dataset + "} \\\\"
    print "\\hline"
    for method in sorted(methods):
        line = []
        if dataset in stats and method in stats[dataset]:
            for x in ordered_stats:
                line.append(stats[dataset][method]['test'][x])
            for x in ordered_stats:
                line.append(stats[dataset][method]['epoch'][x])
        else:
            line.append("missing")
        print make_line(method,line,[],False)

print """
\\hline
\\end{tabular}
\\end{table}
"""
