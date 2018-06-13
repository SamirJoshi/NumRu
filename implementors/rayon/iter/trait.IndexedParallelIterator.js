(function() {var implementors = {};
implementors["ndarray_parallel"] = [{text:"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"rayon/iter/trait.IndexedParallelIterator.html\" title=\"trait rayon::iter::IndexedParallelIterator\">IndexedParallelIterator</a> for <a class=\"struct\" href=\"ndarray_parallel/struct.Parallel.html\" title=\"struct ndarray_parallel::Parallel\">Parallel</a>&lt;AxisIter&lt;'a, A, D&gt;&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/dimension/dimension_trait/trait.Dimension.html\" title=\"trait ndarray::dimension::dimension_trait::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>,&nbsp;</span>",synthetic:false,types:["ndarray_parallel::par::Parallel"]},{text:"impl&lt;'a, A, D&gt; <a class=\"trait\" href=\"rayon/iter/trait.IndexedParallelIterator.html\" title=\"trait rayon::iter::IndexedParallelIterator\">IndexedParallelIterator</a> for <a class=\"struct\" href=\"ndarray_parallel/struct.Parallel.html\" title=\"struct ndarray_parallel::Parallel\">Parallel</a>&lt;AxisIterMut&lt;'a, A, D&gt;&gt; <span class=\"where fmt-newline\">where<br>&nbsp;&nbsp;&nbsp;&nbsp;D: <a class=\"trait\" href=\"ndarray/dimension/dimension_trait/trait.Dimension.html\" title=\"trait ndarray::dimension::dimension_trait::Dimension\">Dimension</a>,<br>&nbsp;&nbsp;&nbsp;&nbsp;A: <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Send.html\" title=\"trait core::marker::Send\">Send</a> + <a class=\"trait\" href=\"https://doc.rust-lang.org/nightly/core/marker/trait.Sync.html\" title=\"trait core::marker::Sync\">Sync</a>,&nbsp;</span>",synthetic:false,types:["ndarray_parallel::par::Parallel"]},];
implementors["rayon"] = [];

            if (window.register_implementors) {
                window.register_implementors(implementors);
            } else {
                window.pending_implementors = implementors;
            }
        
})()
