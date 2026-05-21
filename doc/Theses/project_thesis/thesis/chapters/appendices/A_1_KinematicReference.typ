// Appendix A.x — Kinematic Reference Data (Link Origins & Reference Points)
#import "../../../../shared/formatting/macros.typ": *
#import "../../../../shared/formatting/acronyms.typ": *
#import "../../../../shared/formatting/template.typ": faps-table, faps-figure

= Appendix: Kinematic Reference Data <appendix:kinematic_reference>

This appendix lists the link origins (LO) and tendon/rod attachment
reference points (RP) of the tensegrity manipulator, measured from the
Fusion 360 #ac("CAD") model. All coordinates are expressed in metres
relative to the ceiling-mounted root frame, with the $z$-axis pointing
downward. Reference points~1–4 are symmetric along the $z x$-plane.

#faps-figure(
  image(
    "../../../assets/figures/reference/tensegrity_cad_reference_points.png",
    width: 90%,
  ),
  caption: [#ac("CAD") reference drawing of the tensegrity manipulator with annotated link origins (LO~0–4) and tendon/rod attachment reference points (RP~1–8). Dimensions are given in millimetres.],
  short-caption: [#ac("CAD") reference points of the tensegrity manipulator],
) <fig:cad_reference_points>

== Link Origins <appendix:link_origins>

All link origins lie on the $z$-axis and coincide with the corresponding
joint positions of the kinematic chain.

#faps-table(
  table(
    columns: (auto, auto, auto),
    align: (center, left, center),
    table.header([*\#*], [*Description*], [*Position $(x,thin y,thin z)$ in m*]),
    [0], [Root link],                                   [$(0.0,thin 0.0,thin 0.0)$],
    [1], [Elbow approximation link + joint],            [$(0.0,thin 0.0,thin -0.430)$],
    [2], [Forearm link],                                [$(0.0,thin 0.0,thin -0.4975)$],
    [3], [Wrist link + joints],                         [$(0.0,thin 0.0,thin -0.836)$],
    [4], [Tool link (#ac("TCP"))],                             [$(0.0,thin 0.0,thin -0.972)$],
  ),
  caption: [Link origins (LO) of the 3-#ac("DoF") tensegrity arm, measured from the root frame.],
  short-caption: [Tensegrity arm link origins],
) <tab:link_origins>

== Reference Points <appendix:reference_points>

Tendon and rod attachment points are symmetric along the $z x$-plane.
Points RP~7 and RP~8 are circular distributions: three tendons are spaced
by $120 degree$ on a circle of the indicated radius, centred on the
$z$-axis at the listed height.

#faps-table(
  table(
    columns: (auto, auto, auto, auto),
    align: (center, left, center, left),
    table.header([*\#*], [*Description*], [*Coordinates / radius (m)*], [*Notes*]),
    [1], [Tendon attachment, upper arm],
      [$(0.0,thin 0.0725,thin -0.34)$],
      [Symmetric along $z x$-plane],
    [2], [Rod attachment, upper arm],
      [$(0.0,thin 0.03,thin -0.36)$],
      [Symmetric along $z x$-plane],
    [3], [Rod attachment, forearm],
      [$(0.0,thin 0.03,thin -0.4975)$],
      [Symmetric along $z x$-plane],
    [4], [Tendon attachment, forearm],
      [$(0.0,thin 0.0725,thin -0.5175)$],
      [Symmetric along $z x$-plane],
    [5], [#ac("IMU") centre, forearm],
      [$(0.03735,thin 0.0,thin -0.7422)$],
      [Forearm sensor frame],
    [6], [#ac("IMU") centre, tool link],
      [$(0.0,thin 0.0,thin -0.93)$],
      [Tool sensor frame],
    [7], [Tendon attachment, upper wrist],
      [centre $(0.0,thin 0.0,thin -0.845)$, $r = 0.025$],
      [3 tendons, $120 degree$ spacing],
    [8], [Tendon attachment, lower wrist],
      [centre $(0.0,thin 0.0,thin -0.912)$, $r = 0.0725$],
      [3 tendons, $120 degree$ spacing],
  ),
  caption: [Tendon and rod attachment reference points (RP) of the tensegrity manipulator.],
  short-caption: [Tensegrity arm reference points],
) <tab:reference_points>
