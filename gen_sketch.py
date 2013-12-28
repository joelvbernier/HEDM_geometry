import sys, os
import numpy as np

from hexrd                import matrixutil as mutil
from hexrd.xrd            import rotations  as rot
from hexrd.xrd.transforms import detectorXYToGvec

d2r = np.pi / 180.
r2d = 180. / np.pi

Xl = np.c_[1, 0, 0].T
Yl = np.c_[0, 1, 0].T
Zl = np.c_[0, 0, 1].T

eVec = Xl.flatten()

chi_b  = 15
ome_b  = 5
rMat_b = np.dot(rot.rotMatOfExpMap(d2r * chi_b * Xl), 
                rot.rotMatOfExpMap(d2r * ome_b * Yl))
bVec   = np.dot(rMat_b, -Zl)
beam_len = 10

tVec_d = np.c_[ -5.0,   3.0, -10.0].T
tVec_s = np.c_[  3.0,   0.2,  -1.5].T
tVec_c = np.c_[  2.0,   0.8,  -0.7].T


rMat_d = rot.rotMatOfExpMap(d2r*21*mutil.unitVector(np.c_[2, 3, 1].T))

chi = 21 * d2r
ome = 67 * d2r
rMat_s = np.dot(rot.rotMatOfExpMap(chi * Xl), 
                rot.rotMatOfExpMap(ome * Yl))

rMat_c = rot.rotMatOfExpMap(d2r*36*mutil.unitVector(np.c_[1, 2, 3].T))

det_size = (6, 4)
det_pt   = (-1.15, 0.875)

def gen_sk(fname, det_size, det_pt, eVec,
           beam_len, bVec, rMat_b, 
           rMat_d, rMat_s, rMat_c,
           tVec_d, tVec_s, tVec_c):
    if isinstance(fname, file):
        fid = f
    else:
        fid = open(fname + '.sk', 'w')
    
    phi_b, n_b = rot.angleAxisOfRotMat(rMat_b)
    phi_d, n_d = rot.angleAxisOfRotMat(rMat_d)
    phi_s, n_s = rot.angleAxisOfRotMat(rMat_s)
    phi_c, n_c = rot.angleAxisOfRotMat(rMat_c)
    
    tVec_c_rot = np.dot(rMat_s, tVec_c).flatten()
    tVec_c_l   = tVec_s.flatten() + tVec_c_rot.flatten()
    
    # for diffracted beam
    Z_d  = np.dot(rMat_d, Zl)
    P1_d = np.c_[det_pt[0], det_pt[1], 0.].T
    P1_l = np.dot(rMat_d, P1_d) + tVec_d
    P3_l = 0.
    
    tTh_Eta, gVec_l = detectorXYToGvec(np.c_[det_pt[0], det_pt[1]],
                                       rMat_d, rMat_s,
                                       tVec_d, tVec_s, tVec_c,
                                       beamVec=bVec.reshape(3, 1), 
                                       etaVec=eVec.reshape(3, 1))
    dHat_l = np.dot(np.eye(3) - 2*np.dot(gVec_l, gVec_l.T), bVec)
    
    tTh_vec = np.cross(bVec.flatten(), dHat_l.flatten())
    tTh     = r2d*np.arccos(np.dot(bVec.T, dHat_l))
    
    # eye_vec  = np.dot(2*np.dot(Yl, Yl.T) - np.eye(3), tVec_d).flatten()
    # look_vec = np.dot(np.eye(3) - 2*np.dot(Yl, Yl.T), tVec_d).flatten()
    eye_vec  = np.r_[15, 5, 10]
    look_vec = np.r_[0, 0, -7]

    axwgt_pt = 1.5
    
    """
    -------------------------------------------------
    PRINT THAT SHIT
    -------------------------------------------------
    """
    print >> fid, '% -*-python-*-\n'
    print >> fid, 'def n_segs 180'
    print >> fid, 'def eye     (%f, %f, %f)' % tuple(eye_vec)
    print >> fid, 'def look_at (%f, %f, %f)' % tuple(look_vec)
    
    print >> fid, 'def b0 (0, 0, %f)' % (beam_len)
    print >> fid, 'def b1 (0, 0, %f)' % (-beam_len)
    print >> fid, 'def bpt (%f, %f, %f)' % tuple(1.25*bVec)
    print >> fid, 'def bptd_p (%f, %f, %f)' % tuple(1.5*bVec.flatten())
    print >> fid, 'def bptd_m (%f, %f, %f)' % tuple(-1.5*bVec.flatten())
    print >> fid, 'def bvp (%f, %f, %f)' % tuple(bVec)
    print >> fid, 'def evp (%f, %f, %f)' % tuple(eVec)

    print >> fid, 'def p1 (0,0,0)'
    print >> fid, 'def d1 (%f, %f, %f)' % tuple(P1_l.flatten())
    print >> fid, 'def d2 (%f, %f, %f)' % tuple(1.25*dHat_l.flatten() + tVec_c_l.flatten())
    print >> fid, 'def t2 (%f, %f, %f)' % tuple(tTh_vec.flatten() + tVec_c_l.flatten())
    print >> fid, 'def g1 (%f, %f, %f)' % tuple(gVec_l.flatten())
    
    print >> fid, '\n% detector'
    print >> fid, 'def tvec_d [%f, %f, %f]' % tuple(tVec_d.flatten())
    print >> fid, 'def tpt_d  (%f, %f, %f)' % tuple(tVec_d.flatten())
    print >> fid, 'def tpt_d_label  (%f, %f, %f)' % tuple(0.5*tVec_d.flatten())
        
    print >> fid, '\n% sample'
    print >> fid, 'def tvec_s [%f, %f, %f]' % tuple(tVec_s.flatten())
    print >> fid, 'def tpt_s  (%f, %f, %f)' % tuple(tVec_s.flatten())
    print >> fid, 'def tpt_s_label (%f, %f, %f)' % tuple(0.5*tVec_s.flatten())
    print >> fid, 'def chi    %f' % (r2d*chi)
    print >> fid, 'def ome    %f' % (r2d*ome)
    
    print >> fid, '\n% crystal'
    print >> fid, 'def tvec_c [%f, %f, %f]' % tuple(tVec_c_rot)
    print >> fid, 'def tpt_c  (%f, %f, %f)' % tuple(tVec_c_l)
    print >> fid, 'def tpt_c_label (%f, %f, %f)' % tuple(0.5*(tVec_c_l.flatten() + tVec_s.flatten()))
    print >> fid, 'def tvec_c_l [%f, %f, %f]' % tuple(tVec_c_l)
    print >> fid, 'def tthvec [%f, %f, %f]' % tuple(tTh_vec)
    print >> fid, 'def tth %f' % (tTh)

    print >> fid, 'def g1_label (%f, %f, %f)' % tuple(tVec_c_l.flatten() + 0.8*gVec_l.flatten())
    print >> fid, 'def b1_label (%f, %f, %f)' % tuple(tVec_c_l.flatten() + 1.2*bVec.flatten())
    print >> fid, 'def e1_label (%f, %f, %f)' % tuple(tVec_c_l.flatten() + 0.8*eVec.flatten())
    print >> fid, 'def chi_label (%f, %f, %f)' % tuple(tVec_s.flatten() + np.r_[0, 1, 0.3])
    print >> fid, 'def ome_label (%f, %f, %f)' % tuple(tVec_s.flatten() + 0.6*np.r_[1, 0, -1])
    print >> fid, 'def axlen 1.5'

    print >> fid, 'def det_ang  %f' % (r2d*phi_d)
    print >> fid, 'def det_axis [%f, %f, %f]' % tuple(n_d)
    print >> fid, 'def det_size_x %f'% (det_size[0])
    print >> fid, 'def det_size_y %f'% (det_size[1])
    
    print >> fid, 'def sam_ang  %f' % (r2d*phi_s)
    print >> fid, 'def sam_axis [%f, %f, %f]' % tuple(n_s)
    
    print >> fid, 'def xtl_ang  %f' % (r2d*phi_c)
    print >> fid, 'def xtl_axis [%f, %f, %f]' % tuple(n_c)
    
    print >> fid, '\n% the lab frame'
    print >> fid, \
        'def lab_frame {                                                   \n' + \
        '    line [linewidth=%fpt,arrows=->]  (p1)(axlen,0,0)              \n' % (axwgt_pt) + \
        '    line [linewidth=%fpt,arrows=->]  (p1)(0,axlen,0)              \n' % (axwgt_pt) + \
        '    line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)              \n' % (axwgt_pt) + \
        '    line [arrows=->,linecolor=cyan, lay=over]  (p1)(evp)          \n' + \
        '    line [arrows=->,linecolor=magenta]  (p1)(bvp)                 \n' + \
        '    special |\uput[d ]#1{$\hat{\mathbf{X}}_l$}                    \n' + \
        '             \uput[u ]#2{$\hat{\mathbf{Y}}_l$}                    \n' + \
        '             \uput[u ]#3{$\hat{\mathbf{Z}}_l$}                    \n' + \
        '             \uput[dl]#4{$\hat{\mathbf{e}}$}                      \n' + \
        '             \uput[u ]#5{$\hat{\mathbf{b}}$}                      \n' + \
        '             \uput[d ]#6{$\mathrm{P}_0$}|                         \n' + \
        '        (axlen,0,0)(0,axlen,0)(0,0,axlen)(1,0,0)(0,0,-1.2)(p1)    \n' + \
        '    % put { rotate(beam_ang, (p1), [beam_axis]) }                 \n' + \
        '    %     { line [linewidth=.2pt,linecolor=blue,linestyle=dashed] \n' + \
        '    %         (b0)(b1) }                                          \n' + \
        '    line [linewidth=.2pt,linecolor=blue,linestyle=dashed] (b0)(b1)\n' + \
        '  }                                                               \n'
    
    print >> fid, '\n% the detector'
    print >> fid, \
        'def detector_frame {                                                                            \n' + \
        '    line [linewidth=%fpt,arrows=<->] (axlen,0,0)(p1)(0,axlen,0)                                 \n' % (axwgt_pt) + \
        '    line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)                                            \n' % (axwgt_pt) + \
        '    special |\uput[d]#1{$\hat{\mathbf{X}}_d$}                                                   \n' + \
        '             \uput[r]#2{$\hat{\mathbf{Y}}_d$}                                                   \n' + \
        '             \uput[l]#3{$\hat{\mathbf{Z}}_d$}|                                                  \n' + \
        '        (axlen,0,0)(0,axlen,0)(0,0,axlen)                                                       \n' + \
        '    polygon [fillcolor=gray, lay=under, linecolor=black]                                        \n' + \
        '        (-0.5*det_size_x, -0.5*det_size_y) ( 0.5*det_size_x, -0.5*det_size_y)                   \n' + \
        '        ( 0.5*det_size_x,  0.5*det_size_y) (-0.5*det_size_x,  0.5*det_size_y)                   \n' + \
        '    line [linewidth=.2pt,linecolor=red,linestyle=dashed] (0, -0.6*det_size_y)(0, 0.6*det_size_y)\n' + \
        '    line [linewidth=.2pt,linecolor=red,linestyle=dashed] (-0.6*det_size_x, 0)(0.6*det_size_x, 0)\n' + \
        '  }                                                                                             \n'
    
    print >> fid, '\n% the sample frame'
    print >> fid, \
        'def sample_frame {                                            \n' + \
        '  line [linewidth=%fpt,arrows=<->] (axlen,0,0)(p1)(0,axlen,0) \n' % (axwgt_pt) + \
        '  line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)            \n' % (axwgt_pt) + \
        '  special |\uput[dr]#1{$\hat{\mathbf{X}}_s$}                  \n' + \
        '           \uput[u ]#2{$\hat{\mathbf{Y}}_s$}                  \n' + \
        '           \uput[dr]#3{$\hat{\mathbf{Z}}_s$}|                 \n' + \
        '      (axlen,0,0)(0,axlen,0)(0,0,axlen)(0,0,-1)               \n' + \
        '  }                                                           \n'
    
    print >> fid, '\n% the crystal frame'
    print >> fid, \
        'def crystal_frame {                                           \n' + \
        '  line [linewidth=%fpt,arrows=<->] (axlen,0,0)(p1)(0,axlen,0) \n' % (axwgt_pt) + \
        '  line [linewidth=%fpt,arrows=->]  (p1)(0,0,axlen)            \n' % (axwgt_pt) + \
        '    special |\uput[r ]#1{$\hat{\mathbf{X}}_c$}                \n' + \
        '             \uput[l ]#2{$\hat{\mathbf{Y}}_c$}                \n' + \
        '             \uput[r ]#3{$\hat{\mathbf{Z}}_c$}|               \n' + \
        '        (axlen,0,0)(0,axlen,0)(0,0,axlen)(0,0,-1)             \n' + \
        '  }                                                           \n'
    
    print >> fid, '\n% transform and place objects'
    print >> fid, \
        'def final_detector {                                                                          \n' + \
        '  put { rotate(det_ang, (p1), [det_axis]) then translate([tvec_d])} {detector_frame}          \n' + \
        '  line [arrows=->,linecolor=red]  (p1)(tpt_d)                                                 \n' + \
        '  special |\uput[dr]#1{$\mathrm{P}_1$}                                                        \n' + \
        '           \uput[ul]#2{$\mathbf{t}_d$}                                                        \n' + \
        '           \uput[l ]#3{$\mathrm{P}_4$}|                                                       \n' + \
        '          (tpt_d)(tpt_d_label)(d1)                                                            \n' + \
        '}                                                                                             \n' + \
        'def final_sample {                                                                            \n' + \
        '  put { rotate(sam_ang, (p1), [sam_axis]) then translate([tvec_s])} {sample_frame}            \n' + \
        '  line [lay=over,arrows=->,linecolor=green]  (p1)(tpt_s)                                      \n' + \
        '  put { translate([tvec_s])}                                                                  \n' + \
        '      { line [lay=over,linewidth=.2pt,linecolor=green,linestyle=dashed]                       \n' + \
        '          (0, -1.1*axlen, 0)(0, 1.1*axlen, 0)                                                 \n' + \
        '        line [lay=over,linewidth=.2pt,linecolor=green,linestyle=dashed]                       \n' + \
        '          (-1.1*axlen, 0, 0)(1.1*axlen, 0, 0)                                                 \n' + \
        '        line [lay=over,linewidth=.2pt,linecolor=green,linestyle=dashed]                       \n' + \
        '          (0, 0, -1.1*axlen)(0, 0, 1.1*axlen) }                                               \n' + \
        '  put { translate([tvec_s]) } {                                                               \n' + \
        '    sweep[linecolor=black,arrows=->]{                                                         \n' + \
        '    n_segs, rotate(chi/n_segs, (p1), [1,0,0])}(0,1,0) }                                       \n' + \
        '  put { rotate(sam_ang, (p1), [sam_axis]) then translate([tvec_s]) } {                        \n' + \
        '    { sweep[lay=over,linecolor=black,arrows=<-]{                                              \n' + \
        '      n_segs, rotate(-ome/n_segs, (p1), [0,1,0])}(1,0,0) }                                    \n' + \
        '    { sweep[fillstyle=none,linecolor=black,linestyle=dashed,linewidth=0.2pt]{                 \n' + \
        '      n_segs<>, rotate(360/n_segs, (p1), [0,1,0])}(0,0,1) } }                                 \n' + \
        '  put { translate([tvec_s]) } {                                                               \n' + \
        '    % { sweep[fillstyle=none,linecolor=black,linestyle=dashed,linewidth=0.2pt]{               \n' + \
        '    %   n_segs<>, rotate(360/n_segs, (p1), [0,1,0])}(0,0,1) }                                 \n' + \
        '    { sweep[fillstyle=none,linecolor=black,linestyle=dashed,linewidth=0.2pt]{                 \n' + \
        '      n_segs<>, rotate(360/n_segs, (p1), [1,0,0])}(0,0,1) } }                                 \n' + \
        '  special |\uput[dl]#1{$\mathrm{P}_2$}                                                        \n' + \
        '           \uput[dr]#2{$\mathbf{t}_s$}                                                        \n' + \
        '           \uput[r ]#3{$\omega$}                                                              \n' + \
        '           \uput[l ]#4{$\chi$}|                                                               \n' + \
        '          (tpt_s)(tpt_s_label)(ome_label)(chi_label)                                          \n' + \
        '}                                                                                             \n' + \
        'def final_crystal {                                                                           \n' + \
        '  put { rotate(sam_ang, (p1), [sam_axis]) then translate([tvec_s])                            \n' + \
        '        then rotate(xtl_ang, (tpt_s), [xtl_axis]) then translate([tvec_c]) } {                \n' + \
        '        {crystal_frame} }                                                                     \n' + \
        '  put { translate([tvec_c_l]) } { line [arrows=->,linecolor=cyan]    (p1)(evp) }              \n' + \
        '  put { translate([tvec_c_l]) } { line [arrows=->,linecolor=magenta] (p1)(bvp) }              \n' + \
        '  put { translate([tvec_c_l]) } { line [arrows=->,linecolor=gray]    (p1)(g1)  }              \n' + \
        '  line [arrows=->,linecolor=blue] (tpt_s)(tpt_c)                                              \n' + \
        '  line [lay=over,arrows=->,linecolor=yellow] (tpt_c)(d1)                                      \n' + \
        '  put { translate([tvec_c_l]) } {                                                             \n' + \
        '      { sweep[linecolor=black,arrows=->]{                                                     \n' + \
        '        n_segs, rotate(tth/n_segs, (p1), [tthvec])}(bpt) }                                    \n' + \
        '      { line[linewidth=0.2pt,linecolor=magenta,linestyle=dashed](bptd_m)(bptd_p)} }           \n' + \
        '  special |\uput[u ]#1{$\hat{\mathbf{G}}_{hkl}$}                                              \n' + \
        '           \uput[r ]#2{$\hat{\mathbf{e}}$}                                                    \n' + \
        '           \uput[r ]#3{$\hat{\mathbf{b}}$}                                                    \n' + \
        '           \uput[dr]#4{$\mathrm{P}_3$}                                                        \n' + \
        '           \uput[ r]#5{$\mathbf{t}_c$}                                                        \n' + \
        '           \uput[ur]#6{$2\\theta$}|                                                           \n' + \
        '           (g1_label)(e1_label)(b1_label)(tpt_c)(tpt_c_label)(d2)                             \n' + \
        '}                                                                                             \n' + \
        '                                                                                              \n' + \
        'put { view((eye), (look_at)) } { {lab_frame} {final_detector} {final_sample} {final_crystal} }\n'
    fid.close()

if __name__ == '__main__':
    argv = sys.argv[1:]
    output_name  = argv[0]
    if len(argv) > 1:
        output_dir = argv[1]
    else:
        output_dir = os.getcwd()
    fname_no_suffix = os.path.join(output_dir, output_name)
    gen_sk(fname_no_suffix, det_size, det_pt, eVec, 
           beam_len, bVec, rMat_b, 
           rMat_d, rMat_s, rMat_c, 
           tVec_d, tVec_s, tVec_c)
    f = open(fname_no_suffix + '.tex', 'w')
    print >> f, \
        '\documentclass[pstricks,border=12pt]{standalone}\n' + \
        '\usepackage{amsmath}                            \n' + \
        '\usepackage{pstricks-add}                       \n' + \
        '\\begin{document}                               \n' + \
        '\input{%s.sk.out}                               \n' % (fname_no_suffix) + \
        '\end{document}                                  \n'
    f.close()
    f = open(os.path.join(output_dir, 'run_sketch.sh'), 'w')
    cmd_str = 'cd %s\n' % (output_dir) + \
              'sketch -o %s.sk.out %s.sk\n' % (fname_no_suffix, fname_no_suffix) + \
              'latex %s.tex\n' % (fname_no_suffix) + \
              'dvips -o %s.ps %s.dvi\n' % (fname_no_suffix, fname_no_suffix) + \
              'dvipdf %s.dvi %s.pdf' % (fname_no_suffix, fname_no_suffix)
    print cmd_str
    print >> f, cmd_str
    f.close()

