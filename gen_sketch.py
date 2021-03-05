import os
import sys

import numpy as np

from hexrd import matrixutil as mutil
from hexrd import rotations as rot
from hexrd.transforms import xf

d2r = np.pi / 180.
r2d = 180. / np.pi

Xl = np.c_[1, 0, 0].T
Yl = np.c_[0, 1, 0].T
Zl = np.c_[0, 0, 1].T

eVec = Xl.flatten()

chi_b = 15
ome_b = 5
rMat_b = np.dot(
    rot.rotMatOfExpMap(d2r * chi_b * Xl),
    rot.rotMatOfExpMap(d2r * ome_b * Yl)
)
bVec = np.dot(rMat_b, -Zl)
beam_len = 10

tVec_d = np.c_[-5.0, 3.0, -10.0].T
tVec_s = np.c_[3.0, 0.2, -1.5].T
tVec_c = np.c_[2.0, 0.8, -0.7].T


rMat_d = rot.rotMatOfExpMap(d2r*21*mutil.unitVector(np.c_[2, 3, 1].T))

chi = 21 * d2r
ome = 67 * d2r
rMat_s = np.dot(rot.rotMatOfExpMap(chi * Xl),
                rot.rotMatOfExpMap(ome * Yl))

rMat_c = rot.rotMatOfExpMap(d2r*36*mutil.unitVector(np.c_[1, 2, 3].T))

det_size = (6, 4)
det_pt = (-1.15, 0.875)


def gen_sk(fname, det_size, det_pt, eVec,
           beam_len, bVec, rMat_b,
           rMat_d, rMat_s, rMat_c,
           tVec_d, tVec_s, tVec_c):
    if isinstance(fname, str):
        fid = open(fname + '.sk', 'w')
    else:
        # assume already a file object
        fid = f

    phi_b, n_b = rot.angleAxisOfRotMat(rMat_b)
    phi_d, n_d = rot.angleAxisOfRotMat(rMat_d)
    phi_s, n_s = rot.angleAxisOfRotMat(rMat_s)
    phi_c, n_c = rot.angleAxisOfRotMat(rMat_c)

    tVec_c_rot = np.dot(rMat_s, tVec_c).flatten()
    tVec_c_l = tVec_s.flatten() + tVec_c_rot.flatten()

    # for diffracted beam
    P1_d = np.c_[det_pt[0], det_pt[1], 0.].T
    P1_l = np.dot(rMat_d, P1_d) + tVec_d

    tTh_Eta, gVec_l = xf.detectorXYToGvec(
        np.c_[det_pt[0], det_pt[1]],
        rMat_d, rMat_s,
        tVec_d, tVec_s, tVec_c,
        beamVec=bVec.reshape(3, 1),
        etaVec=eVec.reshape(3, 1)
    )
    dHat_l = np.dot(np.eye(3) - 2*np.dot(gVec_l, gVec_l.T), bVec)

    tTh_vec = np.cross(bVec.flatten(), dHat_l.flatten())
    tTh = r2d*np.arccos(np.dot(bVec.T, dHat_l))

    # eye_vec = np.dot(2*np.dot(Yl, Yl.T) - np.eye(3), tVec_d).flatten()
    # look_vec = np.dot(np.eye(3) - 2*np.dot(Yl, Yl.T), tVec_d).flatten()
    eye_vec = np.r_[15, 5, 10]
    look_vec = np.r_[0, 0, -7]

    axwgt_pt = 1.5

    # -------------------------------------------------
    # %% PRINT THAT SHIT
    # -------------------------------------------------

    block = """
%% -*-python-*-

def n_segs 180
def eye     (%f, %f, %f)
def look_at (%f, %f, %f)
"""
    fill_args = tuple(eye_vec) + tuple(look_vec)
    print(block % fill_args, file=fid)

    block = """
def b0 (0, 0, %f)
def b1 (0, 0, %f)
def bpt (%f, %f, %f)
def bptd_p (%f, %f, %f)
def bptd_m (%f, %f, %f)
def bvp (%f, %f, %f)
def evp (%f, %f, %f)
"""
    fill_args = (beam_len, -beam_len) + tuple(1.25*bVec) + \
        tuple(1.5*bVec.flatten()) + tuple(-1.5*bVec.flatten()) + \
        tuple(bVec) + tuple(eVec)
    print(block % fill_args, file=fid)

    block = """
    def p1 (0, 0, 0)
    def d1 (%f, %f, %f)
    def d2 (%f, %f, %f)
    def t2 (%f, %f, %f)
    def g1 (%f, %f, %f)
    """
    fill_args = tuple(P1_l.flatten()) + \
        tuple(1.25*dHat_l.flatten() + tVec_c_l.flatten()) + \
        tuple(tTh_vec.flatten() + tVec_c_l.flatten()) + \
        tuple(gVec_l.flatten())
    print(block % fill_args, file=fid)

    block = """
%% detector
def tvec_d [%f, %f, %f]
def tpt_d  (%f, %f, %f)
def tpt_d_label  (%f, %f, %f)
"""
    fill_args = tuple(tVec_d.flatten()) + \
        tuple(tVec_d.flatten()) + \
        tuple(0.5*tVec_d.flatten())
    print(block % fill_args, file=fid)

    block = """
%% sample
def tvec_s [%f, %f, %f]
def tpt_s  (%f, %f, %f)
def tpt_s_label (%f, %f, %f)
def chi    %f
def ome    %f
"""
    fill_args = tuple(tVec_s.flatten()) + \
        tuple(tVec_s.flatten()) + \
        tuple(0.5*tVec_s.flatten()) + \
        (r2d*chi, r2d*ome)
    print(block % fill_args, file=fid)

    block = """
%% crystal
def tvec_c [%f, %f, %f]
def tpt_c  (%f, %f, %f)
def tpt_c_label (%f, %f, %f)
def tvec_c_l [%f, %f, %f]
def tthvec [%f, %f, %f]
def tth %f
"""
    fill_args = tuple(tVec_c_rot) + \
        tuple(tVec_c_l) + \
        tuple(0.5*(tVec_c_l.flatten() + tVec_s.flatten())) + \
        tuple(tVec_c_l) + \
        tuple(tTh_vec) + \
        (float(tTh), )
    print(block % fill_args, file=fid)

    block = """
def g1_label (%f, %f, %f)
def b1_label (%f, %f, %f)
def e1_label (%f, %f, %f)
def chi_label (%f, %f, %f)
def ome_label (%f, %f, %f)
def axlen 1.5
"""
    fill_args = tuple(tVec_c_l.flatten() + 0.8*gVec_l.flatten()) + \
        tuple(tVec_c_l.flatten() + 1.2*bVec.flatten()) + \
        tuple(tVec_c_l.flatten() + 0.8*eVec.flatten()) + \
        tuple(tVec_s.flatten() + np.r_[0, 1, 0.3]) + \
        tuple(tVec_s.flatten() + 0.6*np.r_[1, 0, -1])
    print(block % fill_args, file=fid)

    block = """
def det_ang  %f
def det_axis [%f, %f, %f]
def det_size_x %f
def det_size_y %f

def sam_ang  %f
def sam_axis [%f, %f, %f]

def xtl_ang  %f
def xtl_axis [%f, %f, %f]
"""
    fill_args = (r2d*phi_d, ) + \
        tuple(n_d) + \
        (det_size[0], det_size[1]) + \
        (r2d*phi_s, ) + \
        tuple(n_s) + \
        (r2d*phi_c, ) + \
        tuple(n_c)
    print(block % fill_args, file=fid)

    block = """
%% the lab frame
def lab_frame {
    line [linewidth=%fpt,arrows=->]  (p1)(axlen,0,0)
    line [linewidth=%fpt,arrows=->]  (p1)(0,axlen,0)
    line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)
    line [arrows=->,linecolor=cyan, lay=over]  (p1)(evp)
    line [arrows=->,linecolor=magenta]  (p1)(bvp)
    special |\\uput[d ]#1{$\\hat{\\mathbf{X}}_l$}
             \\uput[u ]#2{$\\hat{\\mathbf{Y}}_l$}
             \\uput[u ]#3{$\\hat{\\mathbf{Z}}_l$}
             \\uput[dl]#4{$\\hat{\\mathbf{e}}$}
             \\uput[u ]#5{$\\hat{\\mathbf{b}}$}
             \\uput[d ]#6{$\\mathrm{P}_0$}|
        (axlen,0,0)(0,axlen,0)(0,0,axlen)(1,0,0)(0,0,-1.2)(p1)
    %% put { rotate(beam_ang, (p1), [beam_axis]) }
    %%     { line [linewidth=.2pt,linecolor=blue,linestyle=dashed]
    %%         (b0)(b1) }
    line [linewidth=.2pt,linecolor=blue,linestyle=dashed] (b0)(b1)
  }
"""
    fill_args = (axwgt_pt, axwgt_pt, axwgt_pt)
    print(block % fill_args, file=fid)

    block = """
%% the detector
def detector_frame {
    line [linewidth=%fpt,arrows=<->] (axlen,0,0)(p1)(0,axlen,0)
    line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)
    special |\\uput[d]#1{$\\hat{\\mathbf{X}}_d$}
             \\uput[r]#2{$\\hat{\\mathbf{Y}}_d$}
             \\uput[l]#3{$\\hat{\\mathbf{Z}}_d$}|
        (axlen,0,0)(0,axlen,0)(0,0,axlen)
    polygon [fillcolor=gray, lay=under, linecolor=black]
        (-0.5*det_size_x, -0.5*det_size_y) ( 0.5*det_size_x, -0.5*det_size_y)
        ( 0.5*det_size_x,  0.5*det_size_y) (-0.5*det_size_x,  0.5*det_size_y)
    line [linewidth=.2pt,linecolor=red,linestyle=dashed] (0, -0.6*det_size_y)(0, 0.6*det_size_y)
    line [linewidth=.2pt,linecolor=red,linestyle=dashed] (-0.6*det_size_x, 0)(0.6*det_size_x, 0)
  }
"""
    fill_args = (axwgt_pt, axwgt_pt)
    print(block % fill_args, file=fid)

    block = """
%% the sample frame
def sample_frame {
  line [linewidth=%fpt,arrows=<->] (axlen,0,0)(p1)(0,axlen,0)
  line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)
  special |\\uput[dr]#1{$\\hat{\\mathbf{X}}_s$}
           \\uput[u ]#2{$\\hat{\\mathbf{Y}}_s$}
           \\uput[dr]#3{$\\hat{\\mathbf{Z}}_s$}|
      (axlen,0,0)(0,axlen,0)(0,0,axlen)(0,0,-1)
  }
"""
    fill_args = (axwgt_pt, axwgt_pt)
    print(block % fill_args, file=fid)

    block = """
%% the crystal frame
def crystal_frame {
  line [linewidth=%fpt,arrows=<->] (axlen,0,0)(p1)(0,axlen,0)
  line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)
    special |\\uput[r ]#1{$\\hat{\\mathbf{X}}_c$}
             \\uput[l ]#2{$\\hat{\\mathbf{Y}}_c$}
             \\uput[r ]#3{$\\hat{\\mathbf{Z}}_c$}|
        (axlen,0,0)(0,axlen,0)(0,0,axlen)(0,0,-1)
  }
"""
    fill_args = (axwgt_pt, axwgt_pt)
    print(block % fill_args, file=fid)

    block = """
%% transform and place objects
def final_detector {
  put { rotate(det_ang, (p1), [det_axis]) then translate([tvec_d])} {detector_frame}
  line [arrows=->,linecolor=red]  (p1)(tpt_d)
  special |\\uput[dr]#1{$\\mathrm{P}_1$}
           \\uput[ul]#2{$\\mathbf{t}_d$}
           \\uput[l ]#3{$\\mathrm{P}_4$}|
          (tpt_d)(tpt_d_label)(d1)
}
def final_sample {
  put { rotate(sam_ang, (p1), [sam_axis]) then translate([tvec_s])} {sample_frame}
  line [lay=over,arrows=->,linecolor=green]  (p1)(tpt_s)
  put { translate([tvec_s])}
      { line [lay=over,linewidth=.2pt,linecolor=green,linestyle=dashed]
          (0, -1.1*axlen, 0)(0, 1.1*axlen, 0)
        line [lay=over,linewidth=.2pt,linecolor=green,linestyle=dashed]
          (-1.1*axlen, 0, 0)(1.1*axlen, 0, 0)
        line [lay=over,linewidth=.2pt,linecolor=green,linestyle=dashed]
          (0, 0, -1.1*axlen)(0, 0, 1.1*axlen) }
  put { translate([tvec_s]) } {
    sweep[linecolor=black,arrows=->]{
    n_segs, rotate(chi/n_segs, (p1), [1,0,0])}(0,1,0) }
  put { rotate(sam_ang, (p1), [sam_axis]) then translate([tvec_s]) } {
    { sweep[lay=over,linecolor=black,arrows=<-]{
      n_segs, rotate(-ome/n_segs, (p1), [0,1,0])}(1,0,0) }
    { sweep[fillstyle=none,linecolor=black,linestyle=dashed,linewidth=0.2pt]{
      n_segs<>, rotate(360/n_segs, (p1), [0,1,0])}(0,0,1) } }
  put { translate([tvec_s]) } {
    %% { sweep[fillstyle=none,linecolor=black,linestyle=dashed,linewidth=0.2pt]{
    %%   n_segs<>, rotate(360/n_segs, (p1), [0,1,0])}(0,0,1) }
    { sweep[fillstyle=none,linecolor=black,linestyle=dashed,linewidth=0.2pt]{
      n_segs<>, rotate(360/n_segs, (p1), [1,0,0])}(0,0,1) } }
  special |\\uput[dl]#1{$\\mathrm{P}_2$}
           \\uput[dr]#2{$\\mathbf{t}_s$}
           \\uput[r ]#3{$\omega$}
           \\uput[l ]#4{$\chi$}|
          (tpt_s)(tpt_s_label)(ome_label)(chi_label)
}
def final_crystal {
  put { rotate(sam_ang, (p1), [sam_axis]) then translate([tvec_s])
        then rotate(xtl_ang, (tpt_s), [xtl_axis]) then translate([tvec_c]) } {
        {crystal_frame} }
  put { translate([tvec_c_l]) } { line [arrows=->,linecolor=cyan]    (p1)(evp) }
  put { translate([tvec_c_l]) } { line [arrows=->,linecolor=magenta] (p1)(bvp) }
  put { translate([tvec_c_l]) } { line [arrows=->,linecolor=gray]    (p1)(g1)  }
  line [arrows=->,linecolor=blue] (tpt_s)(tpt_c)
  line [lay=over,arrows=->,linecolor=yellow] (tpt_c)(d1)
  put { translate([tvec_c_l]) } {
      { sweep[linecolor=black,arrows=->]{
        n_segs, rotate(tth/n_segs, (p1), [tthvec])}(bpt) }
      { line[linewidth=0.2pt,linecolor=magenta,linestyle=dashed](bptd_m)(bptd_p)} }
  special |\\uput[u ]#1{$\\hat{\\mathbf{G}}$}
           \\uput[r ]#2{$\\hat{\\mathbf{e}}$}
           \\uput[r ]#3{$\\hat{\\mathbf{b}}$}
           \\uput[dr]#4{$\\mathrm{P}_3$}
           \\uput[ r]#5{$\\mathbf{t}_c$}
           \\uput[ur]#6{$2\\theta$}|
           (g1_label)(e1_label)(b1_label)(tpt_c)(tpt_c_label)(d2)
}

put { view((eye), (look_at)) } { {lab_frame} {final_detector} {final_sample} {final_crystal} }
"""
    print(block, file=fid)

    fid.close()


if __name__ == '__main__':
    argv = sys.argv[1:]
    output_name = argv[0]

    if len(argv) > 1:
        output_dir = argv[1]
    else:
        output_dir = os.getcwd()

    fname_no_suffix = os.path.join(output_dir, output_name)

    gen_sk(fname_no_suffix, det_size, det_pt, eVec,
           beam_len, bVec, rMat_b,
           rMat_d, rMat_s, rMat_c,
           tVec_d, tVec_s, tVec_c)

    with open(fname_no_suffix + '.tex', 'w') as f:
        preamble_str = """
\\documentclass[pstricks,border=12pt]{standalone}
\\usepackage{amsmath}
\\usepackage{pstricks-add}
\\begin{document}
\\input{%s.sk.out}
\\end{document}
"""
        print(preamble_str % (fname_no_suffix), file=f)

    with open(os.path.join(output_dir, 'run_sketch.sh'), 'w') as f:
        cmd_str = """
cd %s
sketch -o %s.sk.out %s.sk
latex %s.tex
dvips -o %s.ps %s.dvi
dvipdf %s.dvi %s.pdf
"""
        fill_args = (
            output_dir,
            fname_no_suffix, fname_no_suffix,
            fname_no_suffix,
            fname_no_suffix, fname_no_suffix,
            fname_no_suffix, fname_no_suffix
        )
        print(cmd_str % fill_args)
        print(cmd_str % fill_args, file=f)
