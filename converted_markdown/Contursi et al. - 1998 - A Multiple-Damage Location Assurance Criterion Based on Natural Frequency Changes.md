A Multiple-Damage Location Assurance Criterion
Based on Natural Frequency Changes
T. CONTURSI
Dipartimento di Progettazione e Producione Industriale, Politecnico di Bari, Bari 70126, Italy
A. MESSINA
Dipartimento di Scienza dei Materiali, Universit&aacute; di Lecce, Lecce 73100, Italy
E. J. WILLIAMS
Department
 of Mechanical Engineering, University of Nottingham, Nottingham, NG7 2RD, UK
(Recieved 17 January 1997; accepted 1 May 1997)
Abstract:
 A new method for locating multiple damage sites in elastic structures is introduced. It uses only the
natural frequencies for diagnosis and obviates the need to have complete experimental mode shapes. Results
from three damaged structures are presented. A 2-bar truss illustrates the basis for the method, a 15-bar truss
shows its use with symmetrical structures, and a redundant-bar truss is used to compare the new method with
that of Pandey and Biswas. Correct predictions of the location and relative amounts of damage at multiple
sites are obtained, even when low damage levels are present in the structure.
Key
 Words: Damage detection, natural frequency, modal parameters
1. INTRODUCTION
During recent years, much work has been dedicated to identifying the location and amount
of damage in structures. Nondestructive evaluation techniques (NDE) offer obvious cost
advantages, safeguard the test structure, and save lives by reducing the chances of unex-
pected collapse.
A number of NDE techniques based on changes in a structure’s modal parameters
have been introduced. One of the first was proposed by Cawley and Adams (1979), who
used the changes in the natural frequencies, together with a finite element (FE) model, to
locate the damage site. Some researchers (Messina, Jones, and Williams, 1996; Penny,
Wilson, and Friswell, 1993) have found this method susceptible to measurement errors,
and ways of improving the localization have been introduced (Williams, Contursi, and
Messina, 1996).
Other authors (Pandey and Biswas, 1995; Zimmermann and Kaouk, 1994; Topole
and Stubbs, 1995) have used the mode shapes in addition to the natural frequencies to
correctly assess the integrity of structures. Pandey and Biswas (1995) used a flexibility
formulation. They claimed that the experimental estimate of the flexibility matrix would
be better than an estimate of the stiffness matrix due to the major contributions of the
Journal of Vibration and Control, 4: 619-633, 1998
@1998 Sage Publications, Inc.
619


620
lowest modes, which are usually easier to measure accurately. Their approach identified
the location and amount of damage in a single pass by calculating the pseudo-inverse of a
matrix to solve a linear system of equations. The authors reported that the accuracy of the
solution depended on the number of mode shapes used in estimating the flexibility matrix,
although they showed that if a reliable numerical model of the structure was available,
then the method could take advantage of it and could perform a damage diagnosis with
fewer modes. This could be taken to imply that the first stage to obtain reliable information
about the health state of the structure should be to obtain an updated FE model, usually
derived in terms of stiffness and mass matrices that accurately predict the measured mode
shapes and natural frequencies. These procedures require considerable care to give reliable
results, and any NDE methods that do not need recourse to an updated model, such as that
proposed in this article, offer considerable attractions.
The algorithm developed by Zimmermann and Kaouk (1994) used two passes-the
first to locate the damage site and the second to assess the amount of damage. It is a robust
numerical code in the presence of measurement errors. However, the need for complete
mode shapes is a requirement that can be difficult to realize in practice.
Topole and Stubbs (1995) also used mode shapes with natural frequencies and showed
the importance of introducing mode shape orthogonality to identify the damaged elements
on a structure by pseudo-inverse solutions of a linear system of equations. In this work,
no assessment of the effects of measurement errors was carried out. The authors identified
two shortcomings of their approach. The first concerns the a priori knowledge of initial
mass and stiffness matrices, and the second is that the method requires measurements at
all degrees of freedom (DOFs) to build complete mode shapes. The latter requirement is
often impossible to meet experimentally due to the inaccessibility of some DOFs. The
use of expansion methods (Berman and Nagy, 1983; Lallement, Ramanitranja, and Cogan,
1996) can assist but, in practice, the ratio of measured to analytical DOFs is invariably low
with consequential errors in the assessment of the mode shapes. This situation is normally
aggravated by the fact that rotational DOFs (which account for 50% of all DOFs) are rarely
measured.
Methods such as the original Cawley and Adams algorithm (1979), which use only
the natural frequencies, therefore retain significant attractions over those that use the mode
shapes as well. While an initial modal survey is required to establish the structure’s mode
shapes to provide a match with the FE model, subsequent damage checks require far fewer
measurements to establish the changes in the natural frequencies. Added to this is the
fact that it has long been recognized that natural frequencies are less contaminated with
measurement errors than either the mode shapes or the damping values.
This article extends the damage location assurance criterion introduced by Messina,
Jones, and Williams (1996) to include the detection of multiple damage sites.
2. DAMAGE LOCATION THEORY
2.1. Damage Location Assurance Criterion
If {0 f is the measured frequency change vector for a structure with damage of unknown
size or location and {<~/)} is the theoretical frequency change vector for damage of a known


621
size at location j, then we can define the damage location assurance criterion (DLAC) for
location j* using a correlation approach similar to the modal assurance criterion (MAC)
value (Ewins, 1984) used for comparing mode shape vectors.
DLAC values lie in the range 0 to 1, with 0 indicating no correlation and 1 indicating
an exact match between the patterns of frequency changes. The location j giving the
highest DLAC value determines the predicted damage site. Messina, Jones, and Williams
(1996) found that a more accurate localization can be obtained if the frequency changes
were normalized with respect to the frequency of the undamaged structures. The use
of percentage changes gives similar weight to all modes, whereas the use of absolute
frequency changes favors high modes since these tend to exhibit larger shifts, as explained
in the appendix. Percentage changes are used throughout the work reported here.
As with the MAC parameter, equation (1) provides a sound statistical measure for
discriminating between the frequency change vectors of potential damage sites. For any
two frequency change vectors, {8 f; and {~//}, a large number of natural frequencies is
desirable from a statistical point of view but is often difficult in practice. This article
analyzes damage in one or more elements in truss structures. For this purpose, it has
been found that only about 10 to 15 modes are required to give sufficient discrimination
to identify which elements are damaged, but not the exact location on each bar. This
information is often sufficient for regular condition checks. One problem with using
higher modes is that the mode shapes can change significantly when damage occurs and
can make it difficult in practice to match the modal pairs from undamaged to damaged
states. The modest mode number requirement makes mode-matching errors less likely and
enhances the value of the approach as a routine condition monitoring tool.
The original DLAC method was shown (Messina, Jones, and Williams, 1996) to be
robust in the presence of error measurements and to give the correct location for damage
at a single site. In particular, it was found that the approach was 100% successful when
detecting the location of a 20% reduction of stiffness at a single site with errors of ~0.3%
in the experimental frequency changes. This level of tolerance to measurement error
compared favorably with laboratory tests that indicated an accuracy for measured natural
frequencies of about ±0.15%. Laboratory tests have confirmed the robustness of the
method in practice (Williams, Messina, and Payne, 1997).
2.2. Sensitivity Matrix
The DLAC formulation uses a database in the form of a matrix in which a typical column,
j, contains {~//}, the frequency change vector due to damage at location j. Where the
number of potential damage sites is small, the data can readily be obtained by rerunning
the FE program with a finite stiffness reduction representing the damage at each site in
turn. This is computationally inefficient and becomes prohibitive for large structures.
An alternative is to use a linear approximation based on derivatives of the eigenvalues
with respect to local stiffness changes. As will be seen later, this approximation is also


622
consistent with the extension of DLAC to multiple-damaged sites. It will be shown that
the required theoretical database can be found by running only one eigensolution.
In this work, each element in the FE model is regarded as a potential damage site, and
damage has been simulated by a homogeneous reduction of stiffness. A stiffness reduction
factor Dj for the j th element is introduced such that Dj = 1 for no damage and Dj = 0
for complete loss of the element (100% damage). The meaning of this reduction in a truss
structure would typically be related to a reduction of the transverse section caused by a
crack or corrosion. The mass matrix has been assumed to remain unchanged.
In this case, the global stiffness matrix [K] can be expressed as a summation of element
matrices, [K]~ of the undamaged structure and their associated stiffness reduction factors
D~ .
where m is the number of elements, and the Boolean assembly matrices [A]j position the
terms from each element matrix within the global matrix such that
Williams, Contursi, and Messina (1996) showed that the sensitivity, ~kj, of the kth
eigenvalue to a small stiffness change in the j th element is
where {~k} is the modal vector for mode k, and [M] is the global mass matrix. The matrix
of sensitivity values has a number of useful properties, as discussed in the appendix.
In terms of the natural frequencies, the eigenvalue sensitivity parameters, Çkj, can be
rewritten as
where fk is the natural frequency (in Hz) of mode k.
2.3. The Multiple-Damage Location Assurance Criterion Approach
For any combination of damage at multiple sites, expressed in terms of changes to individual
stiffness reduction factors, the natural frequency changes can be written using a first-order
approximation as
.’


623
or
or
By measuring the actual frequency changes in a damaged structure, it might appear
from equation (6) that an immediate estimate of the damage state could be obtained directly
by taking a pseudo-inverse solution of the sensitivity matrix.
This results in inaccurate predictions if the stiffness reduction is more than a few
percent. The problem depends first on the linear assumption used and second on the free
domain in which equation (7) looks for a solution. Indeed, a solution via a pseudo-inverse,
as given by equation (7), is not constrained in the correct domain of [0, 1] in which the
stiffness reduction factors represent physical damage to the structure. This can result in the
prediction of significant increases in stiffness that are not physically possible, whereas even
the best approximate solution (e.g., Penrose,1956) makes them mathematically admissible.
Equation (6) gives the frequency changes, {8 f }, resulting from any given pattern of
damage defined by the vector {8D}. Substituting this into equation (1), we obtain a statis-
tical correlation with the measured frequency changes, {0 f }. This is termed the multiple-
damage location assurance criterion (MDLAC) since it is a function of all elements in the
damage vector {8D}.
The required assessment of the structure’s damage state is obtained by searching for the
vector {8D}, which maximizes the MDLAC value. This can be expressed mathematically
in the form


624
Figure 1. Two-bar truss.
Table 1. Damaged state scenarios for the two-bar truss.
The search for the maximum in {<SD} 
space is a computationally intensive exercise,
although, as will be illustrated in the cases studied, the search range can be restricted
without loss of generality.
3. NUMERICAL SIMULATIONS
All of the assessments have been done by using the high-level language of the MATLAB
package. An FE package (also written using MATLAB) has been used to carry out eigen-
solutions of the various truss structures studied. The stiffness and mass matrix for the
elements were built with a classical consistent formulation without internal nodes (Petyt,
1990).
3.1. A Two-Bar Truss Structure
A two-bar truss system with two modes is used to provide a graphical illustration of the
ability of the MDLAC to discriminate between different damage conditions. Figure 1
shows the structure with its material and geometrical properties, and Table 1 gives the four
damage scenarios presented. In each case, stiffness reductions of the amounts shown were


625
Figure 2. Surface plot of the MDLAC parameter for each damage scenario for the two-bar truss.
applied to the structure, and the new natural frequencies of the damaged structure were
calculated to give the &dquo;experimental&dquo; frequency change vector, {Af 1.
Figure 2 shows surface plots of the MDLAC parameter for values of 8 D and<$D2 
in the
range from 0% (no damage) to 100% (complete loss of the element). Since the MDLAC
is a pattern recognition function based on a linear approximation, it cannot predict the
absolute amount of damage but does correctly predict the proportion of damage in each
bar of the truss. For example, with scenario 2 (see Figure 2b), the MDLAC has a maximum
for all damage states for which 6 Di : 8D2 is 4:3, as shown by the thick line on the figure.
This agrees with the actual ratio of damage. The same is true with the other scenarios.
If the starting value for the maximum search is the undamaged state, as used here, it
is not necessary to search the whole domain of the stiffness reduction factors between 0
and 1. A search domain restricted to a maximum reduction of 0.5 (SO% damage) has been
found to be effective in reducing the computational effort involved.
A further computational advantage can be obtained by raising the MDLAC value to a
power. While the maximum value remains 1.0, the gradients of the search function near
to the maximum are increased, so speeding convergence. This is illustrated in Figure 3
with exponents of 20 and 200. Although a high exponent would seem to be more effective,
numerical problems were met in the search algorithms if values in excess of 20 were used.


626
Figure 3. Surface plot of the MDLAC parameter with different exponents for scenario 2.
As already stated, while the MDLAC can identify the proportion of damage at one
or more locations, it provides no information about the absolute damage level. This
same problem was present in Cawley and Adams’s (1979) algorithm. Further work is
required, probably involving another pass, to address this problem. In practice, however,
it is probably more important to have information about the location(s) of damage than an
estimate of the size and, in this respect, the approach is successful.
. 
I


627
Figure 4. Fifteen-bar truss structure and its mode shapes.
Table 2. Damaged state scenarios for the 15-bar truss.
3.2. Symmetrical 15-Bar Truss
Figure 4 shows a symmetrical truss structure with its material and geometrical properties,
and Table 2 gives six damage state scenarios. Figure 5 shows the results of each case
using the first 12 natural frequencies of the structure. The {<5D} vector, which maximizes
the MDLAC function, is presented and normalized so that the largest value is unity. The
actual maximum MDLAC value is shown in the legend in each case.
Figures 5a and 5b both indicate two damage sites, whereas only one site is actually
damaged in each scenario. In each case, the algorithm correctly identifies the actual
damage site but assigns an equal probability to damage being present in the corresponding
symmetrically placed member.


628
Figure 5. Results of MDLAC approach on a symmetrical 15-bar truss.
Figures 5c, 5d, and 5e examine different amounts of damage in bar 12. It will be seen
from the legends that the maximum value of the MDLAC increases (from 0.77 to 0.92 to
1.0) as the amount of damage decreases, indicating that the method works better when a
low level of damage is present. This is a consequence of the linear approximation inherent
in the use of the sensitivity matrix in the formulation. Nevertheless, it is clear that the
algorithm is able to predict damaged sites correctly even when the damage level is not
low. This contrasts with the attempted direct solution to the problem given by equation
(8), which gave inaccurate predictions if the stiffness reduction was more than about 2%.
Figure 5f shows the case of equal amounts of damage at three sites-two on symmetric
members (4 and 10) and one on the nonsymmetric bar 7. It will be seen that the relative
levels of damage are correctly identified, bearing in mind that half of the actual damage in
bar 4 is &dquo;shared&dquo; with its symmetric pair, bar 5. The same thing happens with bar 10.
3.3. Redundant 51-Bar Truss
The MDLAC method has been compared with the approach of Pandey and Biswas (1995),
using one of their test structures (depicted in Figure 6).


629
Figure 6. Ten-bay, 51-bar truss.
Table 3. Damaged state scenarios for the 10-bay, 51-bar truss.
Figure 7 compares the results of the two methods for the damage scenarios in Table 3.
Following the recommendations in Pandey and Biswas’s (1995) article, 10 complete mode
shapes and 10 natural frequencies were used, together with the exact flexibility matrix of
the undamaged structure. The changes in the first 10 natural frequencies were used in the
MDLAC calculations, with the sensitivity matrix computed from equation (7). As with
the symmetrical 15-bar truss, the maximum search was initiated with values of 10-4 for
all stiffness reduction factors to simulate the initial undamaged state.
Figure 7a shows that both methods correctly predict the damage state for scenario 1.
However, Pandey and Biswas’s (1995) method also indicates a number of other bars with
significant changes, including some with stiffness increases. The situation is repeated in
scenario 2, shown in Figure 7b, in which lower levels of damage are present. Pandey and
Biswas’s (1995) method correctly indicates the location and size of the damage, but the
false indications remain. This could lead to uncertainty in a blind test, as would be met in
practice.
In scenario 3, shown in Figure 7c, both methods identify the true locations but fail to
give correct estimates of the relative amounts of damage present. Even so, the MDLAC
predictions are more accurate.
While both methods correctly identify the location and relative size of the damage in
scenario 4, shown in Figure 7d, both also give false indications. With Pandey and Biswas’s
(1995) method, this is again in the fonn of stiffness increases at the locations observed in
other scenarios. In the case of the MDLAC approach, damage is also indicated in bars 25
and 39, the latter being the &dquo;symmetric&dquo; pair to the damaged bar 44.
While Pandey and Biswas’s (1995) method is able to provide an assessment of the
damage level, it has the disadvantages of giving more false indications and of requiring


630
Figure 7. Comparison of MDLAC and Pandey & Biswas’s Method.
complete mode shape information to calculate changes to the flexibility matrix. In practice,
the latter would require either a comprehensive modal survey each time a damage evaluation
is needed or the use of a model expansion technique to obtain the required data. Either
way, it is felt that the MDLAC approach offers the more cost-effective solution.
4. DISCUSSION OF RESULTS
In each of the three cases presented, it is found that the MDLAC approach gives good
predictions of the location of both single- and multiple-damage sites and of the relative
amount of damage at each.
, 
_ 
.


631
The method is attractive for practical applications since it only requires information
about changes in the structure’s natural frequencies. Unlike many other methods, it gives
good predictions for low levels of damage. This is due to the use of a sensitivity matrix based
on small perturbations from the undamaged state. It should be remembered, however, that
measurement errors due to the need to detect small frequency changes could limit its use.
At higher levels of damage, Figures 5c, 5d, and 5e show that some uncertainty is
introduced due to the actual nonlinear relationship between frequency changes and damage.
Even so, the highest MDLAC values are still found at the true damage locations. The
problem could be reduced by replacing the derivative-based sensitivity matrix by one based
on finite differences in which the frequency changes due to a finite stiffness reduction (e.g.,
20%) at each potential damage site in turn. However, this would mean that the sensitivities
could no longer be obtained from the computationally efficient equation (4).
The issue of the effect of the linear sensitivity model raises the wider matter of struc-
tural modeling errors. These have not been addressed explicitly here and are the subject of
ongoing investigation. However, Williams, Messina, and Payne (1997) used the sensitivity
model of a steel frame to predict damage in a fiber-reinforced composite component. The
two structures were geometrically identical, but there were gross differences between mate-
rial properties. However, although the measured natural frequencies differed considerably
from the analytical values, the use of percentage frequency changes in the formulation
allowed the method to successfully predict the correct damage site despite these modeling
errors.
Previous work (Messina, Jones, and Williams, 1996; Williams, Messina, and Payne,
1997) also suggests that the method can readily tolerate the error levels typical of laboratory-
based natural frequency measurements. While no formal assessment of measurement
errors is included here, it should be noted that the solution frequency change vectors,
{8f ({8D})}, are not exact matches to the &dquo;measured&dquo; vectors, {Af 1. This is apparent, for
example, in Figure 5c, where the maximum MDLAC value is just 0.77. Despite this, the
approach correctly identifies both the locations and the relative size of the damage present.
It would be reasonable to expect similar success in the presence of actual measurement
errors.
The method also involves a computationally demanding search for the maximum. For
example, the 51-bar, 10-bay truss example with 10 modes takes about 12 minutes to run
on a 75 MHz Pentium PC. While significant, it remains small in comparison with the time
required to collect and process the data to obtain the frequency estimates.
5. CONCLUSIONS
A new correlation coefficient termed the MDLAC has been shown to provide reliable
information about the location and relative size of damage at one or more sites on a
structure. It has the practical attraction of only requiring information about the changes in
the natural frequencies between the undamaged and damaged states.
The matrix used in the formulation for describing the sensitivity of each natural fre-
quency to small reductions in local stiffness from the undamaged state can be obtained
efficiently using equation (4) and requires only one eigensolution of an FE model of the


632
structure. This sensitivity matrix has useful properties (see the appendix) that help to verify
the data.
APPENDIX
Properties of the Sensitivity Matrix
The sensitivity expression in equation (4) has two interesting properties.
Property 1
Any element of the sensitivity matrix, [S], is a positive-semidefinite term. Indeed, ~kj
or, more correctly, the numerator in equation (4), corresponds at a positive-semidefinite
quadratic form, {x }T [ K~ ] {x }, of [ K~ ], the local stiffness matrix for element j . This property
implies that a local reduction in the Dj coefficient results in a decrease of any natural
frequency, as required from physical considerations.
Property 2
The sum of all elements in the kth row of [S] is equal at the eigenvalue Àk. This can be
seen by considering the classical eigenvalue problem for the undamaged structure.
Rearranging and premultiplying for {~k}T, we obtain
Using equation (2) in (A2) and comparing the result with equation (4), we obtain
The first property is consistent with physical stiffness changes to the system. The two
properties together could be used to check the accuracy of the implementation of sensitivity
matrix assessments in an FE package.
Moreover, from equation (A3), it is possible to show the value of using percentage
frequency changes instead of absolute frequency changes in the DLAC or MDLAC for-
mulations. Using equation (5), we can write
Considering the independence of the index k with respect to j, it is found that


633
and finally,
It can be seen that the mean value of each row of the natural frequency sensitivity
matrix is proportional to the corresponding natural frequency. Thus, for a given level of
damage, the absolute frequency changes will tend to increase with mode number, whereas
the percentage changes will show greater mode-to-mode consistency. This observation
explains why both the DLAC and MDLAC formulations give better results using percentage
frequency change data.
REFERENCES
Berman, A. and Nagy, E. J.,1983, "Improvement of a large analytical model using test data," AIAA
 
Journal 
21
(8), 1168-1173.
Cawley, P. and Adams, R. D., 1979, "The location of defects in structures from measurements of natural frequencies,"
Journal of Strain Analysis 14(2), 49-57.
Ewins, D. J., 1984, Modal Testing: Theory and Practice, Research Studies Press, Somerset, UK.
Lallement, G., Ramamtranja, A., and Cogan, M., 1996, "Optimal sensors deployment: Applications to model updating
problems," in Proceedings of the Identification in Engineering Systems, Swansea, Wales, pp. 338-356.
Messina, A., Jones, I. A., and Williams, E. J., 1996, "Damage detection and localisation using natural frequency changes,"
in Proceedings of the Identification in Engineering Systems, Swansea, Wales, pp. 67-76.
Pandey, A. K. and Biswas, M., 1995, "Damage diagnosis of truss structures by estimation of flexibility change," Modal
Analysis: The International Journal of Analytical and Experimental Modal Analysis 10
(2), 104-117.
Penny, J.E.T., Wilson, D., and Friswell, M. I., 1993, "Damage location in structures using vibration data," in Proceedings
of the 11 th International Modal Analysis Conference, Kissimee, FL, pp. 861-867.
Penrose, R.,1956, "On best approximate solutions of linear matrix equations," 
Proceedings of the Cambridge Philosophical
Society 52, 17-19.
Petyt, M., 1990, Introduction to Finite Element Vibration Analysis, Cambridge University Press, Cambridge, UK.
Topole, K. G. and Stubbs, N., 1995, "Nondestructive damage evaluation in complex structures from a minimum of modal
parameters," Modal Analysis: The
 International Journal
 of Analytical and Experimental Modal Analysis 10(2), 95-
103.
Williams, E. J., Contursi, T., and Messina, A., 1996, "Damage detection and localisation using natural frequency sensitivity,"
in Proceedings of the Identification in Engineering Systems, Swansea, Wales, pp. 368-376.
Williams, E. J., Messina, A., and Payne, B. S., 1997, "A frequency-change correlation approach to damage detection," in
Proceedings of the 15th International Modal Analysis Conference, Orlando, FL, pp. 652-657.
Zimmermann, D. C. and Kaouk, M., 1994, "Structural damage detection using a minimum rank update theory," Journal of
Vibration and Acoustics 116, 222-231.


