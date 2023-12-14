$('#grid-item').click(function() {
   $('form').toggle('slow');
});

$(document).ready(function() {
    var currentIndex = 0;
    var items = $('.carousel-item');
    var itemAmount = items.length;

    $('.carousel-next').click(function() {
        if (currentIndex < itemAmount - 1) {
            currentIndex++;
            $('.carousel-inner').css({
                'transform': 'translateX(' + (-currentIndex * 100) + '%)'
            });
        }
    });

    $('.carousel-prev').click(function() {
        if (currentIndex > 0) {
            currentIndex--;
            $('.carousel-inner').css({
                'transform': 'translateX(' + (-currentIndex * 100) + '%)'
            });
        }
    });
});